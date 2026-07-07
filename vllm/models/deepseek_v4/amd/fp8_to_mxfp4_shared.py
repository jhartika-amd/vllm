# SPDX-License-Identifier: Apache-2.0
"""WS2 — load-time FP8(block-scale) -> MXFP4 re-quant of DSv4's shared expert.

DSv4-Flash routes experts in **MXFP4** (fp4x2 + E8M0 per-1x32 microscale) but
stores the always-on shared expert as **FP8** (E4M3 + 128x128 block scale). To
fold the shared expert into the routed grouped-GEMM (mechanism B, router-append)
its weights must be converted to the exact MXFP4 checkpoint layout the routed
expert slots expect, then loaded into slots ``[n_routed .. n_routed+n_shared)``.

Pipeline (per projection weight ``W`` of shape ``[out, in]``):
    FP8 e4m3 + block scale  --dequant-->  bf16[out,in]
    bf16                    --mxfp4----->  (packed uint8[out,in//2], e8m0[out,in//32])

The MXFP4 quantizer is aiter's canonical weight quantizer ``dynamic_mxfp4_quant``
(per-1x32 block, unshuffled) — the same OCP microscaling format the routed
checkpoint ships, so everything downstream (the expert ``weight_loader``,
``process_weights_after_loading`` -> ``convert_weight_to_mxfp4_moe_kernel_format``
swizzle/gfx942 repack) is reused unchanged.

Kept as a separate module so it can be unit-tested in isolation (WS2-T3) without
standing up the full model — see ``_test_fp8_to_mxfp4_shared.py``.
"""

from __future__ import annotations

import torch


def dequant_fp8_block(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Dequantize a DeepSeek block-quantized FP8 weight to bf16.

    ``weight``           : [out, in]  FP8 (e4m3)
    ``weight_scale_inv`` : [ceil(out/bn), ceil(in/bk)] fp32 block multipliers
    Returns bf16 [out, in]  ==  weight.float() * expand(weight_scale_inv).
    """
    assert weight.dim() == 2, f"expected 2D weight, got {tuple(weight.shape)}"
    out, inn = weight.shape
    bn, bk = block
    w = weight.to(torch.float32)
    scale = weight_scale_inv.to(torch.float32)
    # Expand the per-block scale to full [out, in] then trim any ragged edge.
    scale = scale.repeat_interleave(bn, dim=0)[:out]
    scale = scale.repeat_interleave(bk, dim=1)[:, :inn]
    return (w * scale).to(torch.bfloat16)


def quant_bf16_to_mxfp4(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [out, in] -> (packed_uint8 [out, in//2], scale_uint8 [out, in//32]).

    Uses aiter's ``dynamic_mxfp4_quant`` (per-1x32 E8M0, unshuffled) and returns
    the packed weight / scale as raw ``uint8`` bytes — the layout the MXFP4
    expert params (``w{13,2}_weight`` / ``w{13,2}_weight_scale``) hold straight
    from the checkpoint, before ``process_weights_after_loading`` swizzles them.
    """
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant

    assert w_bf16.dim() == 2
    # ``dynamic_mxfp4_quant`` is a Triton kernel and needs a CUDA tensor, but
    # weights arrive on CPU during loading. Quantize on GPU, then return the
    # bytes on the input's original device so the expert weight_loader sees the
    # same device it gets for the routed-expert checkpoint tensors.
    orig_device = w_bf16.device
    w = w_bf16.contiguous()
    if w.device.type != "cuda":
        w = w.cuda()
    packed, scale = dynamic_mxfp4_quant(w, shuffle=False)
    # dtype-view to raw bytes; the expert weight_loader / DSv4 loader treat the
    # MXFP4 weight and E8M0 scale as uint8 (see model.py e8m0 view handling).
    packed = packed.view(torch.uint8).to(orig_device).contiguous()
    scale = scale.view(torch.uint8).to(orig_device).contiguous()
    return packed, scale


def convert_shared_fp8_to_mxfp4(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 block-scale [out, in] -> MXFP4 (packed uint8, scale uint8).

    Convenience wrapper: dequant then MXFP4-quant. This is the single call the
    weight loader makes per shared-expert projection (w1/w2/w3) before handing
    the result to the routed-expert ``weight_loader`` for slot ``n_routed+j``.
    """
    return quant_bf16_to_mxfp4(dequant_fp8_block(weight, weight_scale_inv, block))
