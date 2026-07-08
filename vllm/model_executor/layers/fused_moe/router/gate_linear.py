# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os as _os

import torch
from torch.nn.parameter import Parameter

import vllm._custom_ops as ops
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


# --- Gate-logits fused cast (Phase 3 Lever-B.3) ----------------------------
# On ROCm the CUDA-only fast tiers (1-3) in GateLinear.forward are all disabled
# (they are gated on current_platform.is_cuda()), so DSv4 falls to Tier 4:
# a bf16 F.linear followed by a SEPARATE `output.to(torch.float32)` cast to make
# the fp32 router logits the sqrtsoftplus gating consumes. That trailing cast is
# the standalone `aten::copy_` (dtype cast bf16->fp32, ~0.22 ms / ~50 launches
# per decode step) attributed to gate_linear.py:forward in the with-stack glue
# profile. Folding it into the GEMM epilogue -- one bf16 x @ bf16 W.T with
# out_dtype=fp32 (rocBLAS accumulates straight to fp32) -- removes the extra
# kernel. This mirrors the CUDA-only Tier 3 (cuBLAS bf16->fp32) but runs on
# ROCm/hipBLAS. Flag-gated so the fold is trivially A/B-testable and falls back
# to the stock Tier 4 path when off.
def _use_gate_fused_cast() -> bool:
    return _os.environ.get("VLLM_DSV4_GATE_FUSED_CAST", "0").lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    """MoE gate linear layer with multi-tier GEMM dispatch:

    1. DSV3 specialized kernel (SM90+, M<=16, H=7168 E=256/384, H=6144 E=256)
    2. fp32 specialized kernel  (SM90+, bf16/fp32 in, fp32 out,
       M<=32, H=3072, E=256)
    3. cuBLAS bf16×bf16→fp32 (SM90+ + bf16 weight + fp32 out_dtype)
    3b. ROCm fused-cast GEMM (bf16×bf16→fp32 via torch.mm out_dtype; folds the
        router-logits dtype cast into the GEMM epilogue). Flag-gated on
        VLLM_DSV4_GATE_FUSED_CAST.
    4. F.linear via ReplicatedLinear (ultimate fallback)

    The ``out_dtype`` attribute is mutable and can be set after init
    (e.g. when the required dtype depends on the expert quantization
    method which is only known later).
    """

    # Dimensions supported by the DSV3 specialized kernel.
    # Valid (hidden_size, num_experts) combinations:
    #   (7168, 256) -> DeepSeek-V3,  (7168, 384) -> Kimi-K2,
    #   (6144, 256) -> GLM-5
    DSV3_SUPPORTED_NUM_EXPERTS = [256, 384]
    DSV3_SUPPORTED_HIDDEN_SIZES = [7168, 6144]
    # num_experts=384 is only instantiated for hidden_size=7168.
    DSV3_UNSUPPORTED_SHAPES = {(6144, 384)}

    # (hidden_size, num_experts) pairs with an instantiated fp32 kernel:
    #   (3072, 256) -> MiniMax-M2/M2.5,  (6144, 128) -> MiniMax-M3
    FP32_SUPPORTED_SHAPES = {(3072, 256), (6144, 128)}
    FP32_MAX_TOKENS = 32

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        is_hopper = current_platform.is_device_capability((9, 0))
        is_blackwell = current_platform.is_device_capability_family(100)
        can_use_specialized_kernels = (
            current_platform.is_cuda() and (is_hopper or is_blackwell) and not bias
        )

        # If fp32 compute is required and no specialized kernel is available,
        # store weights in fp32 so the fallback linear path computes in fp32.
        if force_fp32_compute and not can_use_specialized_kernels:
            params_dtype = torch.float32

        super().__init__(
            input_size,
            output_size,
            bias=bias,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        self.out_dtype = out_dtype

        # DSV3 specialized kernel eligibility (SM90+, exact dims)
        self.allow_specialized_router_gemm = can_use_specialized_kernels
        self.allow_dsv3_router_gemm = (
            self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and output_size in self.DSV3_SUPPORTED_NUM_EXPERTS
            and input_size in self.DSV3_SUPPORTED_HIDDEN_SIZES
            and (input_size, output_size) not in self.DSV3_UNSUPPORTED_SHAPES
        )
        # See https://github.com/vllm-project/vllm/pull/44217
        # for more details.
        self._dsv3_max_batch = 16 if is_hopper else 8

        # fp32 specialized kernel eligibility (SM90+, exact dims, fp32 weight)
        self.allow_fp32_router_gemm = (
            not bias
            and self.weight.dtype == torch.float32
            and current_platform.is_cuda()
            and (is_hopper or is_blackwell)
            and (input_size, output_size) in self.FP32_SUPPORTED_SHAPES
        )

        # cuBLAS bf16→fp32 eligibility
        self.allow_cublas_router_gemm = (
            self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

        # ROCm fused-cast GEMM eligibility (prototype, Phase 3 Lever-B.3):
        # bf16 weight + fp32 out_dtype + no bias, and the CUDA fast tiers are
        # unavailable (i.e. we would otherwise fall to Tier 4 + a trailing cast).
        # The runtime x.dtype == bf16 check stays in forward().
        self.allow_rocm_fused_cast_router_gemm = (
            not bias
            and not self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

    def set_out_dtype(self, out_dtype: torch.dtype) -> None:
        """Set output dtype for the router logits after init.

        Useful when the required dtype depends on the expert quantization
        method which is only known after the gate is constructed.
        """
        if self.out_dtype is not None:
            raise ValueError("out_dtype has already been set")
        self.out_dtype = out_dtype

        if (
            not self.allow_cublas_router_gemm
            and self.allow_specialized_router_gemm
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16

        # Keep the ROCm fused-cast eligibility in sync when out_dtype is set late.
        if (
            not self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and out_dtype == torch.float32
        ):
            self.allow_rocm_fused_cast_router_gemm = True

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # Tier 1: DSV3 specialized kernel
        if self.allow_dsv3_router_gemm and x.shape[0] <= self._dsv3_max_batch:
            output = ops.dsv3_router_gemm(
                hidden_states=x,
                router_weight=self.weight,
                output_dtype=self.out_dtype,
            )
            return output, None

        # Tier 2: fp32 specialized kernel (H=3072, E=256, M<=32)
        # Dispatch is wrapped in a custom op so that torch.compile/CUDA-graph
        # capture does not freeze the runtime num_tokens branch.
        if self.allow_fp32_router_gemm and x.dtype in (
            torch.float32,
            torch.bfloat16,
        ):
            output = torch.ops.vllm.fp32_router_gemm_dispatch(x, self.weight)
            return output, None

        # Tier 3: cuBLAS bf16→fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
            output = torch.mm(x, self.weight.T, out_dtype=torch.float32)
            return output, None

        # Tier 3b (prototype): ROCm fused-cast GEMM. Fold the router-logits
        # bf16->fp32 cast into the GEMM epilogue (single rocBLAS bf16 x @ bf16
        # W.T accumulating to fp32), removing the standalone Tier 4 `output.to(
        # fp32)` copy (~0.22 ms / decode step). Flag-gated; identical math to
        # Tier 4 (fp32-accumulated bf16 matmul), just without the extra kernel.
        if (
            _use_gate_fused_cast()
            and self.allow_rocm_fused_cast_router_gemm
            and x.dtype == torch.bfloat16
        ):
            output = torch.mm(x, self.weight.T, out_dtype=torch.float32)
            return output, None

        # Tier 4: F.linear (ReplicatedLinear)
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias


_FP32_ROUTER_GEMM_MAX_TOKENS = GateLinear.FP32_MAX_TOKENS


def fp32_router_gemm_dispatch_impl(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    Dynamically run fp32 specialized gemm if num_tokens <= FP32_MAX_TOKENS,
    otherwise fall back to F.linear.
    This must be wrapped in a custom op because our torch.compile integration
    does not support runtime dispatching on num_tokens.
    """
    if x.shape[0] <= _FP32_ROUTER_GEMM_MAX_TOKENS:
        return ops.fp32_router_gemm(x, weight)
    else:
        return torch.nn.functional.linear(x.float(), weight)


def fp32_router_gemm_dispatch_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="fp32_router_gemm_dispatch",
    op_func=fp32_router_gemm_dispatch_impl,
    fake_impl=fp32_router_gemm_dispatch_fake,
)
