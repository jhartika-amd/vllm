# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import time
from dataclasses import dataclass
from typing import Any, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
    DeepseekV4FlashMLAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
    rocm_inv_rope_einsum,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
)
from vllm.logger import init_logger
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)
_KV_CACHE_LAYOUT_LOGGED = False


def _maybe_log_kv_cache_layout(
    layer_prefix: str,
    swa_k_cache: torch.Tensor,
    compressed_k_cache: torch.Tensor | None,
    kv_cache_dtype: str,
) -> None:
    global _KV_CACHE_LAYOUT_LOGGED
    if _KV_CACHE_LAYOUT_LOGGED or os.environ.get("VLLM_DSV4_LOG_KV_CACHE_LAYOUT") != "1":
        return
    if swa_k_cache.numel() == 0:
        return
    _KV_CACHE_LAYOUT_LOGGED = True
    flat512 = _flydsl_prefill_cache_is_flat512(swa_k_cache)
    msg = (
        f"[dsv4-kv-layout] layer={layer_prefix} kv_cache_dtype={kv_cache_dtype} "
        f"swa shape={tuple(swa_k_cache.shape)} dtype={swa_k_cache.dtype} "
        f"swa_stride0={swa_k_cache.stride(0)} flat512={flat512}"
    )
    if compressed_k_cache is not None and compressed_k_cache.numel() > 0:
        msg += (
            f" compressed shape={tuple(compressed_k_cache.shape)} "
            f"dtype={compressed_k_cache.dtype} stride0={compressed_k_cache.stride(0)}"
        )
    logger.info(msg)


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.int32).contiguous()
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


# --- FlyDSL gfx942 sparse-MLA prefill (WS2, flag-gated) --------------------
# Routes the DSv4 prefill attention off the Triton (dequant-gather -> merged
# CSR -> ragged sparse-flash over a bf16 KV workspace) pipeline and onto the
# aiter branch `samremes/gfx942-sparse-mla-prefill` FlyDSL kernel
# `flydsl_sparse_mla_prefill_2region`, which reads the two packed fp8_ds_mla
# caches directly (native fp8 MFMA). Region roles are positional:
#   region0 / main_*  = SWA cache  (always-present window -> non-empty per
#                       query, satisfies the kernel's region0 contract),
#                       main_is_fnuz=True on gfx942, main_scale_mode='ue8m0'.
#   region1 / extra_* = compressed top-k cache (OCP -> extra_is_fnuz=False,
#                       extra_block_size = block_size // compress_ratio).
# rope_bf16=True matches vLLM's NoPE-fp8 / RoPE-bf16 numerics (cos > 0.999).
# Packaged as a standalone site or resolved from the image's aiter; import
# failure / flag off / swa-only falls back to the stock Triton path.
# The T>=32768 int32 shape-packing overflow is fixed upstream (aiter commit
# `T>=32768 support added with test`, per-CTA int64 buffer base), so no query
# sub-tiling is needed. Gate: VLLM_DSV4_FLYDSL_PREFILL.
_FLYDSL_PREFILL_FN = None
_FLYDSL_PREFILL_IMPORT_TRIED = False


def _flydsl_prefill_fn():
    global _FLYDSL_PREFILL_FN, _FLYDSL_PREFILL_IMPORT_TRIED
    if not _FLYDSL_PREFILL_IMPORT_TRIED:
        _FLYDSL_PREFILL_IMPORT_TRIED = True
        fn = None
        try:
            from aiter.ops.flydsl import flydsl_sparse_mla_prefill_2region as fn
        except Exception:
            try:
                from flydsl_sparse_mla import flydsl_sparse_mla_prefill_2region as fn
            except Exception:
                fn = None
        _FLYDSL_PREFILL_FN = fn
    return _FLYDSL_PREFILL_FN


def _use_flydsl_prefill() -> bool:
    if os.environ.get("VLLM_DSV4_FLYDSL_PREFILL", "0").lower() in {
        "0",
        "false",
        "no",
        "off",
        "",
    }:
        return False
    return _flydsl_prefill_fn() is not None


def _flydsl_prefill_cache_is_flat512(cache: torch.Tensor) -> bool:
    return cache.dim() == 3 and cache.shape[-1] == 512


_PREFILL_TRACE_PATH = os.environ.get(
    "VLLM_DSV4_FLYDSL_PREFILL_TRACE_PATH", "/tmp/flydsl_prefill_trace.jsonl"
)
_PREFILL_TRACE_COUNT = 0
_PREFILL_TRACE_DUMP_DIR = os.environ.get(
    "VLLM_DSV4_FLYDSL_PREFILL_TRACE_DUMP_DIR", "/tmp/flydsl_prefill_dumps"
)


def _flydsl_prefill_trace_enabled() -> bool:
    return os.environ.get("VLLM_DSV4_FLYDSL_PREFILL_TRACE", "0") == "1"


def _flydsl_prefill_trace_max() -> int:
    return int(os.environ.get("VLLM_DSV4_FLYDSL_PREFILL_TRACE_MAX", "0"))


def _flydsl_prefill_trace_sync() -> bool:
    return os.environ.get("VLLM_DSV4_FLYDSL_PREFILL_TRACE_SYNC", "0") == "1"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    return str(obj)


def _flydsl_prefill_trace_rank0() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank()) == 0
    except Exception:
        return True


def _check_region_slots(
    *,
    region: str,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    q_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_blocks: int,
    block_stride_bytes: int,
    row_bytes: int,
    cache_numel_bytes: int,
    seq_lens: torch.Tensor | None = None,
) -> list[str]:
    """Host-side mirror of kernel slot -> block_table -> byte offset addressing."""
    issues: list[str] = []
    num_queries = int(indptr.shape[0] - 1)
    max_blocks = int(block_table.shape[1])
    indptr_cpu = indptr.detach().cpu()
    idx_cpu = indices.detach().cpu()
    q_req_cpu = q_req.detach().cpu()
    bt_cpu = block_table.detach().cpu()
    seq_cpu = seq_lens.detach().cpu() if seq_lens is not None else None

    for q in range(num_queries):
        req = int(q_req_cpu[q].item())
        if req < 0 or req >= block_table.shape[0]:
            issues.append(f"{region} q={q} bad q_req={req} bt_rows={block_table.shape[0]}")
            continue
        start = int(indptr_cpu[q].item())
        end = int(indptr_cpu[q + 1].item())
        if end <= start:
            # Empty region1 (compressed top-k) is normal for early prefill positions.
            if region != "extra":
                issues.append(f"{region} q={q} req={req} empty segment [{start},{end})")
            continue
        seg = idx_cpu[start:end]
        seq_len = int(seq_cpu[req].item()) if seq_cpu is not None else None
        for slot in seg.tolist():
            if slot < 0:
                if region == "extra":
                    continue
                issues.append(f"{region} q={q} req={req} negative slot={slot}")
                continue
            if seq_len is not None and slot >= seq_len:
                issues.append(
                    f"{region} q={q} req={req} slot={slot} >= seq_len={seq_len}"
                )
            block_idx = slot // block_size
            pos = slot - block_idx * block_size
            if block_idx < 0 or block_idx >= max_blocks:
                issues.append(
                    f"{region} q={q} req={req} slot={slot} block_idx={block_idx} "
                    f">= max_blocks={max_blocks}"
                )
                continue
            phys = int(bt_cpu[req, block_idx].item())
            if phys < 0:
                issues.append(
                    f"{region} q={q} req={req} slot={slot} phys={phys} (block_idx={block_idx})"
                )
                continue
            if phys >= num_blocks:
                issues.append(
                    f"{region} q={q} req={req} slot={slot} phys={phys} >= num_blocks={num_blocks}"
                )
            byte_off = phys * block_stride_bytes + pos * row_bytes
            byte_end = byte_off + row_bytes
            if byte_off < 0 or byte_end > cache_numel_bytes:
                issues.append(
                    f"{region} q={q} req={req} slot={slot} phys={phys} pos={pos} "
                    f"bytes=[{byte_off},{byte_end}) cache_bytes={cache_numel_bytes} "
                    f"stride={block_stride_bytes}"
                )
    return issues


def _trace_flydsl_prefill_call(
    *,
    chunk_idx: int,
    chunk_size: int,
    query_start: int,
    query_end: int,
    seq_lens: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_indptr: torch.Tensor,
    q_req: torch.Tensor,
    main_block_table: torch.Tensor,
    extra_block_table: torch.Tensor,
    swa_k_cache: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    block_size: int,
    extra_block_size: int,
    N: int,
    flydsl_kw: dict[str, Any],
) -> None:
    global _PREFILL_TRACE_COUNT
    if not _flydsl_prefill_trace_enabled():
        return
    if not _flydsl_prefill_trace_rank0():
        return
    trace_max = _flydsl_prefill_trace_max()
    if trace_max > 0 and _PREFILL_TRACE_COUNT >= trace_max:
        return

    flat512 = _flydsl_prefill_cache_is_flat512(swa_k_cache)
    swa_num_blocks = int(swa_k_cache.shape[0])
    swa_stride_b = int(swa_k_cache.stride(0) * swa_k_cache.element_size())
    swa_cache_bytes = int(swa_k_cache.untyped_storage().nbytes())
    extra_num_blocks = int(compressed_k_cache.shape[0])
    extra_stride_b = int(compressed_k_cache.stride(0) * compressed_k_cache.element_size())
    extra_cache_bytes = int(compressed_k_cache.untyped_storage().nbytes())
    row_bytes = 512

    main_issues = _check_region_slots(
        region="main",
        indices=main_indices,
        indptr=main_indptr,
        q_req=q_req,
        block_table=main_block_table,
        block_size=block_size,
        num_blocks=swa_num_blocks,
        block_stride_bytes=swa_stride_b,
        row_bytes=row_bytes,
        cache_numel_bytes=swa_cache_bytes,
        seq_lens=seq_lens,
    )
    extra_issues = _check_region_slots(
        region="extra",
        indices=extra_indices,
        indptr=extra_indptr,
        q_req=q_req,
        block_table=extra_block_table,
        block_size=extra_block_size,
        num_blocks=extra_num_blocks,
        block_stride_bytes=extra_stride_b,
        row_bytes=row_bytes,
        cache_numel_bytes=extra_cache_bytes,
        seq_lens=None,
    )
    # Compressed top-k rows must stay within N (CSR builder contract).
    extra_nnz = int(extra_indptr[-1].item())
    if extra_nnz > 0:
        extra_pos = extra_indices[:extra_nnz]
        extra_pos = extra_pos[extra_pos >= 0]
        if extra_pos.numel() > 0:
            emax = int(extra_pos.max().item())
            emin = int(extra_pos.min().item())
            if emax >= N:
                extra_issues.append(f"extra max index {emax} >= N={N}")
        else:
            emin = emax = -1
    else:
        emin = emax = -1

    main_pos = main_indices[main_indices >= 0]
    safe_kw = {
        k: (
            {"shape": list(v.shape), "dtype": str(v.dtype)}
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in flydsl_kw.items()
    }
    record: dict[str, Any] = {
        "ts": time.time(),
        "call": _PREFILL_TRACE_COUNT,
        "chunk_idx": chunk_idx,
        "chunk_size": chunk_size,
        "num_queries": int(main_indptr.shape[0] - 1),
        "num_tokens": int(query_end) - int(query_start),
        "query_range": [query_start, query_end],
        "seq_lens": seq_lens.detach().cpu().tolist(),
        "N": N,
        "flat512": bool(flat512),
        "swa_cache": {
            "shape": list(swa_k_cache.shape),
            "stride0": int(swa_k_cache.stride(0)),
            "num_blocks": swa_num_blocks,
            "block_stride_bytes": swa_stride_b,
        },
        "extra_cache": {
            "shape": list(compressed_k_cache.shape),
            "stride0": int(compressed_k_cache.stride(0)),
            "num_blocks": extra_num_blocks,
            "block_stride_bytes": extra_stride_b,
        },
        "main_bt_shape": list(main_block_table.shape),
        "extra_bt_shape": list(extra_block_table.shape),
        "q_req_minmax": [
            int(q_req.min().item()),
            int(q_req.max().item()),
        ],
        "main_slot_minmax": [
            int(main_pos.min().item()) if main_pos.numel() else -1,
            int(main_pos.max().item()) if main_pos.numel() else -1,
        ],
        "extra_slot_minmax": [emin, emax],
        "main_nnz": int(main_indices.shape[0]),
        "extra_nnz": int(extra_indices.shape[0]),
        "flydsl_kw": safe_kw,
        "issues": main_issues + extra_issues,
    }
    _PREFILL_TRACE_COUNT += 1
    suspect = bool(record["issues"])
    record["suspect"] = suspect

    os.makedirs(os.path.dirname(_PREFILL_TRACE_PATH) or ".", exist_ok=True)
    with open(_PREFILL_TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")

    if suspect:
        logger.error(
            "[flydsl-prefill-trace] suspect call #%s chunk=%s issues=%s",
            record["call"],
            chunk_idx,
            record["issues"][:8],
        )
        os.makedirs(_PREFILL_TRACE_DUMP_DIR, exist_ok=True)
        dump_path = os.path.join(
            _PREFILL_TRACE_DUMP_DIR, f"call_{record['call']:05d}_chunk_{chunk_idx}.pt"
        )
        torch.save(
            {
                "record": record,
                "main_indices": main_indices.detach().cpu(),
                "main_indptr": main_indptr.detach().cpu(),
                "extra_indices": extra_indices.detach().cpu(),
                "extra_indptr": extra_indptr.detach().cpu(),
                "q_req": q_req.detach().cpu(),
                "main_block_table": main_block_table.detach().cpu(),
                "extra_block_table": extra_block_table.detach().cpu(),
                "seq_lens": seq_lens.detach().cpu(),
            },
            dump_path,
        )
        logger.error("[flydsl-prefill-trace] dumped tensors to %s", dump_path)


@triton.jit
def _build_two_region_csr_kernel(
    pos_ptr,  # [num_tokens] int32, absolute seq position of each query token
    main_indptr_ptr,  # [num_tokens + 1] int32 (region0 / SWA)
    extra_indptr_ptr,  # [num_tokens + 1] int32 (region1 / compressed)
    topk_indices_ptr,
    topk_indices_stride,
    main_indices_ptr,  # [main_nnz] int32 out
    extra_indices_ptr,  # [extra_nnz] int32 out
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_WINDOW: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    token_idx = tl.program_id(0)
    pos = tl.load(pos_ptr + token_idx)
    swa_len = tl.minimum(pos + 1, WINDOW_SIZE)
    topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)

    # region0 (SWA): request-local absolute positions [pos-swa_len+1 .. pos].
    # These are the SAME positions the dequant kernel walks; the FlyDSL kernel
    # maps them through main_block_table (the SWA block table) + q_req.
    # arange over next-pow2(window) (Triton requires pow2) + mask to swa_len.
    m_start = tl.load(main_indptr_ptr + token_idx)
    swa_offset = tl.arange(0, PADDED_WINDOW)
    swa_mask = swa_offset < swa_len
    tl.store(
        main_indices_ptr + m_start + swa_offset,
        pos - swa_len + 1 + swa_offset,
        mask=swa_mask,
    )

    # region1 (compressed top-k): request-local compressed rows, validated to
    # [0, N); out-of-range -> -1 (the kernel masks negative / >= skv slots).
    e_start = tl.load(extra_indptr_ptr + token_idx)
    topk_offset = tl.arange(0, PADDED_TOP_K)
    topk_mask = topk_offset < topk_len
    safe_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
    idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + safe_offset,
        mask=topk_mask,
        other=-1,
    )
    valid = (idx >= 0) & (idx < N)
    idx = tl.where(valid, idx, -1)
    tl.store(extra_indices_ptr + e_start + topk_offset, idx, mask=topk_mask)


def build_two_region_prefill_csrs(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    top_k: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the two per-query ragged CSRs the FlyDSL 2-region kernel wants.

    Returns (main_indices, main_indptr, extra_indices, extra_indptr, q_req).
    ``query_start_loc`` is the chunk's [num_reqs + 1] cumulative query offsets
    (any base; rebased internally). Lengths are deterministic from the absolute
    position, so indptr is a host cumsum and a single Triton pass fills values.
    """
    device = topk_indices.device
    q_start = query_start_loc.to(torch.int64)
    base = q_start[0]
    q_start = q_start - base
    q_lens = (q_start[1:] - q_start[:-1]).to(torch.int32)
    num_reqs = int(q_lens.shape[0])
    num_tokens = int(q_start[-1].item())

    req_ids = torch.arange(num_reqs, device=device, dtype=torch.int64)
    q_req_i64 = torch.repeat_interleave(req_ids, q_lens.to(torch.int64))
    tok_in_req = (
        torch.arange(num_tokens, device=device, dtype=torch.int64)
        - q_start[:-1][q_req_i64]
    )
    start_pos = seq_lens.to(torch.int64) - q_lens.to(torch.int64)
    pos = (start_pos[q_req_i64] + tok_in_req).to(torch.int32)

    swa_len = torch.clamp(pos + 1, max=window_size)
    topk_len = torch.clamp(
        torch.div(pos + 1, compress_ratio, rounding_mode="floor"), max=top_k
    )
    main_indptr = _build_indptr_from_lengths(swa_len)
    extra_indptr = _build_indptr_from_lengths(topk_len)
    main_nnz = int(main_indptr[-1].item())
    extra_nnz = int(extra_indptr[-1].item())

    main_indices = torch.empty(main_nnz, dtype=torch.int32, device=device)
    # Kernel needs a valid pointer even when region1 is globally empty; zero so
    # unused slots never carry uninitialized values into validation or the GPU.
    extra_indices = torch.zeros(max(extra_nnz, 1), dtype=torch.int32, device=device)

    topk_indices = topk_indices.reshape(num_tokens, -1).contiguous()
    _build_two_region_csr_kernel[(num_tokens,)](
        pos,
        main_indptr,
        extra_indptr,
        topk_indices,
        topk_indices.stride(0),
        main_indices,
        extra_indices,
        N,
        TOP_K=top_k,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        PADDED_WINDOW=triton.next_power_of_2(window_size),
        TOPK_WIDTH=topk_indices.shape[-1],
        PADDED_TOP_K=triton.next_power_of_2(max(topk_indices.shape[-1], 1)),
    )
    return main_indices, main_indptr, extra_indices, extra_indptr, q_req_i64.to(
        torch.int32
    )


# ROCm sparse prefill keeps this dense combine local so AMD-specific SWA changes
# do not touch the shared DeepSeek V4 cache utilities.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        topk_offset = tl.arange(0, PADDED_TOP_K)
        topk_mask = topk_offset < topk_len
        safe_topk_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + safe_topk_offset,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = (topk_indices >= 0) & (topk_indices < N)
        topk_indices = tl.where(valid_topk, topk_indices + M * batch_idx, -1)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offset,
            topk_indices,
            mask=topk_mask,
        )

        swa_offset = tl.arange(0, WINDOW_SIZE)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offset,
            M * batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start,
            mask=swa_offset < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    num_workers = 128
    _combine_topk_swa_indices_kernel[(num_reqs, num_workers)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        TOPK_WIDTH=topk_indices.shape[-1],
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _compute_topk_lens_kernel(
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        count += tl.sum((local_idx >= 0).to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@triton.jit
def _pack_global_topk_ragged_kernel(
    global_topk_ragged_ptr,
    topk_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
    if block_idx * BLOCK_SIZE >= out_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    mask = (offset < out_len) & (offset < topk)
    local_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    valid = mask & (local_idx >= 0)
    block_indices = local_idx // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid,
        other=0,
    )
    block_offsets = local_idx % block_size
    slot_ids = tl.where(valid, block_numbers * block_size + block_offsets, -1)
    tl.store(global_topk_ragged_ptr + out_start + offset, slot_ids, mask=mask)


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


@triton.jit
def _compute_combined_lens_kernel(
    combined_lens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)
        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


@triton.jit
def _combine_topk_swa_indices_ragged_kernel(
    combined_ragged_ptr,
    combined_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    topk_width,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    block_idx = tl.program_id(2)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)
        combined_len = topk_len + swa_len

        offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if block_idx * BLOCK_SIZE < combined_len:
            out_start = tl.load(combined_indptr_ptr + token_idx)
            topk_mask = (offset < topk_len) & (offset < topk_width)
            topk_vals = tl.load(
                topk_indices_ptr + token_idx * topk_indices_stride + offset,
                mask=topk_mask,
                other=-1,
            )
            tl.store(
                combined_ragged_ptr + out_start + offset,
                topk_vals + M * batch_idx,
                mask=topk_mask,
            )

            swa_offset = offset - topk_len
            swa_mask = (offset >= topk_len) & (swa_offset < swa_len)
            tl.store(
                combined_ragged_ptr + out_start + offset,
                M * batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start,
                mask=swa_mask,
            )


def combine_topk_swa_indices_ragged(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    num_workers = 128
    _compute_combined_lens_kernel[(num_reqs, num_workers)](
        combined_lens,
        query_start_loc,
        seq_lens,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
    )

    combined_indptr = _build_indptr_from_lengths(combined_lens)
    combined_ragged = torch.empty(
        num_tokens * (topk + window_size),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if combined_ragged.numel() > 0:
        block = 128
        _combine_topk_swa_indices_ragged_kernel[
            (num_reqs, num_workers, triton.cdiv(topk + window_size, block))
        ](
            combined_ragged,
            combined_indptr,
            topk_indices,
            topk_indices.stride(0),
            query_start_loc,
            seq_lens,
            gather_lens,
            M,
            N,
            topk_indices.shape[-1],
            TOP_K=topk,
            COMPRESS_RATIO=compress_ratio,
            WINDOW_SIZE=window_size,
            BLOCK_SIZE=block,
        )
    return combined_ragged, combined_indptr, combined_lens


def _copy_ragged_to_graph_buffers(
    ragged_indices: torch.Tensor,
    ragged_indptr: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
    num_rows: int,
    max_entries_per_row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy dynamic ragged metadata into persistent CUDA graph buffers.

    FULL decode graphs capture kernel argument addresses. Keep the returned
    tensors backed by stable storage, while indptr continues to bound reads.
    """
    indptr_out = ragged_indptr_buffer[: num_rows + 1]
    indptr_out.copy_(ragged_indptr, non_blocking=True)

    max_entries = max(num_rows * max_entries_per_row, 1)
    ragged_out = ragged_indices_buffer[:max_entries]
    nnz = ragged_indices.numel()
    if nnz > 0:
        ragged_out[:nnz].copy_(ragged_indices, non_blocking=True)
    return ragged_out, indptr_out


@dataclass
class DeepseekV4ROCMAiterMLASparseMetadata(DeepseekV4FlashMLAMetadata):
    """ROCm-specific DeepSeek V4 metadata carrying ragged decode topk."""

    c128a_decode_topk_ragged_indices: torch.Tensor | None = None
    c128a_decode_topk_ragged_indptr: torch.Tensor | None = None


@dataclass
class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None


class DeepseekV4ROCMAiterMLASparseMetadataBuilder(DeepseekV4FlashMLAMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c128a_decode_topk_ragged_indices_buffer: torch.Tensor | None = None
        self.c128a_decode_topk_ragged_indptr_buffer: torch.Tensor | None = None
        if self.compress_ratio == 128:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            self.c128a_decode_topk_ragged_indices_buffer = torch.empty(
                max_tokens * self.c128a_max_compressed,
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_decode_topk_ragged_indptr_buffer = torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterMLASparseMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        dense_decode = base.c128a_global_decode_topk_indices
        decode_lens = base.c128a_decode_topk_lens
        if dense_decode is not None and decode_lens is not None:
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                dense_decode.reshape(dense_decode.shape[0], -1),
                decode_lens,
            )
            assert self.c128a_decode_topk_ragged_indices_buffer is not None
            assert self.c128a_decode_topk_ragged_indptr_buffer is not None
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.c128a_decode_topk_ragged_indices_buffer,
                self.c128a_decode_topk_ragged_indptr_buffer,
                dense_decode.shape[0],
                self.c128a_max_compressed,
            )

        return DeepseekV4ROCMAiterMLASparseMetadata(
            **vars(base),
            c128a_decode_topk_ragged_indices=ragged_indices,
            c128a_decode_topk_ragged_indptr=ragged_indptr,
        )


class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.decode_swa_ragged_indices_buffer = torch.empty(
            max_tokens * self.window_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_ragged_indptr_buffer = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterSparseSWAMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        if (
            base.num_decode_tokens > 0
            and base.decode_swa_indices is not None
            and base.decode_swa_lens is not None
        ):
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                base.decode_swa_indices.reshape(base.num_decode_tokens, -1),
                base.decode_swa_lens,
            )
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.decode_swa_ragged_indices_buffer,
                self.decode_swa_ragged_indptr_buffer,
                base.num_decode_tokens,
                self.window_size,
            )

        return DeepseekV4ROCMAiterSparseSWAMetadata(
            **vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr,
        )


class DeepseekV4ROCMAiterMLASparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterMLASparseMetadataBuilder"]:
        return DeepseekV4ROCMAiterMLASparseMetadataBuilder


class DeepseekV4ROCMAiterMLAAttention(DeepseekV4Attention):
    """ROCm sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4ROCMAiterMLASparseBackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # ROCm BF16 reference wo_a path (inverse RoPE + einsum) + wo_b.
        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        return self.wo_b(z.flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        rocm_metadata = cast(
            DeepseekV4ROCMAiterMLASparseMetadata | None,
            attn_metadata.get(self.prefix),
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache
        _maybe_log_kv_cache_layout(
            self.prefix,
            swa_kv_cache,
            self_kv_cache,
            getattr(self, "kv_cache_dtype", "?"),
        )

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=rocm_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=rocm_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                (
                    topk_ragged_indices,
                    topk_ragged_indptr,
                    topk_lens,
                ) = compute_global_topk_ragged_indices_and_indptr(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
                topk_ragged_indices = attn_metadata.c128a_decode_topk_ragged_indices
                topk_ragged_indptr = attn_metadata.c128a_decode_topk_ragged_indptr

        rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
            self.PREFILL_CHUNK_SIZE
        )

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only and _use_flydsl_prefill():
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                query_start = (
                    query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
                )
                query_end = (
                    query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
                )
                qsl_chunk = query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ]
                (
                    main_indices,
                    main_indptr,
                    extra_indices,
                    extra_indptr,
                    q_req,
                ) = build_two_region_prefill_csrs(
                    topk_indices[query_start:query_end],
                    qsl_chunk,
                    seq_lens[chunk_start:chunk_end],
                    self.window_size,
                    self.compress_ratio,
                    top_k,
                    N,
                )
                swa_block_table = swa_metadata.block_table[num_decodes:]
                compressed_block_table = attn_metadata.block_table[num_decodes:]
                flat512 = _flydsl_prefill_cache_is_flat512(swa_k_cache)
                flydsl_kw: dict = {
                    "block_size": swa_metadata.block_size,
                    "attn_sink": self.attn_sink,
                    "extra_block_size": attn_metadata.block_size // self.compress_ratio,
                    "q_req": q_req,
                    "main_is_fnuz": current_platform.is_fp8_fnuz(),
                    "extra_is_fnuz": False,
                    "single_request": False,
                    "validate_regions": True,
                }
                if flat512:
                    flydsl_kw["main_scale_mode"] = "per_tensor"
                    flydsl_kw["rope_bf16"] = False
                    flydsl_kw["q_scale"] = getattr(
                        self, "_flashinfer_fp8_q_scale", None
                    )
                    flydsl_kw["main_kv_scale"] = getattr(
                        self.swa_cache_layer, "_flashinfer_fp8_kv_scale", None
                    )
                    flydsl_kw["extra_kv_scale"] = getattr(
                        self, "_flashinfer_fp8_kv_scale", None
                    )
                else:
                    flydsl_kw["main_scale_mode"] = "ue8m0"
                    flydsl_kw["rope_bf16"] = True
                _trace_flydsl_prefill_call(
                    chunk_idx=chunk_idx,
                    chunk_size=chunk_size,
                    query_start=query_start,
                    query_end=query_end,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    main_indices=main_indices,
                    main_indptr=main_indptr,
                    extra_indices=extra_indices,
                    extra_indptr=extra_indptr,
                    q_req=q_req,
                    main_block_table=swa_block_table[chunk_start:chunk_end],
                    extra_block_table=compressed_block_table[chunk_start:chunk_end],
                    swa_k_cache=swa_k_cache,
                    compressed_k_cache=compressed_k_cache,
                    block_size=swa_metadata.block_size,
                    extra_block_size=attn_metadata.block_size // self.compress_ratio,
                    N=N,
                    flydsl_kw=flydsl_kw,
                )
                _flydsl_prefill_fn()(
                    q[query_start:query_end],
                    output[query_start:query_end],
                    swa_k_cache,
                    main_indices,
                    main_indptr,
                    swa_block_table[chunk_start:chunk_end],
                    compressed_k_cache,
                    extra_indices,
                    extra_indptr,
                    compressed_block_table[chunk_start:chunk_end],
                    **flydsl_kw,
                )
                if _flydsl_prefill_trace_sync():
                    torch.cuda.synchronize()
                continue
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                # compressed_k_cache is OCP on every platform (Triton encoder).
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            rocm_sparse_attn_prefill(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices,
                topk_length=combined_lens,
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                output=output[query_start:query_end],
            )
