# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-channel Power Spectral Density (PSD) matrix.

Aligned with ``torchaudio.functional.psd``.  The Triton kernel fuses the
channel-wise outer product, the time-dimension reduction and (optional) mask
weighting into a single pass, so the large intermediate tensor
``(..., freq, time, channel, channel)`` produced by a naive ``torch.einsum`` is
never materialized.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _psd_kernel(
    spec_real_ptr,
    spec_imag_ptr,
    mask_ptr,
    out_real_ptr,
    out_imag_ptr,
    channel,
    time,
    spec_stride_b,
    spec_stride_c,
    spec_stride_f,
    spec_stride_t,
    mask_stride_b,
    mask_stride_f,
    mask_stride_t,
    out_stride_b,
    out_stride_f,
    out_stride_c1,
    out_stride_c2,
    eps,
    HAS_MASK: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """One program per (batch, freq).

    Accumulates the ``channel x channel`` outer product ``X X^H`` over the
    time axis entirely in registers; the intermediate ``time x channel x
    channel`` tensor is never written back to global memory.
    """
    pid = tl.program_id(0)
    # freq is the innermost index so that adjacent programs read adjacent
    # frequency bins (coalesced along the time stride).
    freq_idx = tl.program_id(1)
    batch_idx = pid

    offs_c = tl.arange(0, BLOCK_C)
    offs_e = tl.arange(0, BLOCK_C)
    c_mask = offs_c < channel
    e_mask = offs_e < channel

    # Accumulate in the input's native real-part precision (fp32 for
    # complex64, fp64 for complex128).  Narrower inputs (complex32) are
    # upcast to complex64 by the wrapper before reaching the kernel.
    acc_dtype = spec_real_ptr.dtype.element_ty
    accum_real = tl.zeros([BLOCK_C, BLOCK_C], dtype=acc_dtype)
    accum_imag = tl.zeros([BLOCK_C, BLOCK_C], dtype=acc_dtype)
    # Scalar accumulator for the mask sum, in the same precision as the
    # outer-product accumulator (avoids a fp32/fp64 type mismatch when the
    # mask is float64).
    mask_sum = tl.sum(tl.zeros([1], dtype=acc_dtype))

    # Base offset for this (batch, freq) in the (batch, channel, freq, time)
    # spectrogram layout.  Strides are expressed in *float* elements because
    # the real/imag views double the complex element stride.
    spec_base = batch_idx * spec_stride_b + freq_idx * spec_stride_f

    for t_start in range(0, time, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < time

        # X[:, t] for the current time block: shape (BLOCK_C, BLOCK_T).
        addr = spec_base + offs_c[:, None] * spec_stride_c + t_offs[None, :] * spec_stride_t
        load_mask = c_mask[:, None] & t_mask[None, :]
        xr = tl.load(spec_real_ptr + addr, mask=load_mask, other=0.0)
        xi = tl.load(spec_imag_ptr + addr, mask=load_mask, other=0.0)

        # Outer product X[c] * conj(X[e]) over the time block.
        #   real = xr[c]*xr[e] + xi[c]*xi[e]
        #   imag = xi[c]*xr[e] - xr[c]*xi[e]
        xr_c = xr[:, None, :]  # (C, 1, T)
        xi_c = xi[:, None, :]
        xr_e = xr[None, :, :]  # (1, C, T)
        xi_e = xi[None, :, :]

        real_3d = xr_c * xr_e + xi_c * xi_e  # (C, C, T)
        imag_3d = xi_c * xr_e - xr_c * xi_e

        if HAS_MASK:
            m_addr = (
                batch_idx * mask_stride_b
                + freq_idx * mask_stride_f
                + t_offs * mask_stride_t
            )
            m_val = tl.load(mask_ptr + m_addr, mask=t_mask, other=0.0).to(acc_dtype)
            mask_sum += tl.sum(m_val)
            real_3d = real_3d * m_val[None, None, :]
            imag_3d = imag_3d * m_val[None, None, :]

        accum_real += tl.sum(real_3d, axis=2)
        accum_imag += tl.sum(imag_3d, axis=2)

    if HAS_MASK and NORMALIZE:
        denom = mask_sum + eps
        accum_real = accum_real / denom
        accum_imag = accum_imag / denom

    # Store (batch, freq, channel, channel).
    out_base = batch_idx * out_stride_b + freq_idx * out_stride_f
    out_addr = out_base + offs_c[:, None] * out_stride_c1 + offs_e[None, :] * out_stride_c2
    out_mask = c_mask[:, None] & e_mask[None, :]
    tl.store(out_real_ptr + out_addr, accum_real.to(out_real_ptr.dtype.element_ty), mask=out_mask)
    tl.store(out_imag_ptr + out_addr, accum_imag.to(out_imag_ptr.dtype.element_ty), mask=out_mask)


def psd(
    specgram: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute cross-channel power spectral density (PSD) matrix.

    Args:
        specgram: Multi-channel complex-valued spectrum.
            Tensor with dimensions ``(..., channel, freq, time)``.
        mask: Time-frequency mask for normalization.
            Tensor with dimensions ``(..., freq, time)``. (Default: ``None``)
        normalize: If ``True``, normalize the mask along the time dimension.
            (Default: ``True``)
        eps: Value added to the denominator in mask normalization.
            (Default: ``1e-10``)

    Returns:
        Complex-valued PSD matrix of the input spectrum.
        Tensor with dimensions ``(..., freq, channel, channel)``.
    """
    if not isinstance(specgram, torch.Tensor):
        raise TypeError(f"specgram must be a torch.Tensor, got {type(specgram)}")
    if specgram.ndim < 3:
        raise ValueError(
            f"specgram must have at least 3 dimensions (channel, freq, time), "
            f"got {specgram.ndim}"
        )
    if not specgram.is_complex():
        raise ValueError(f"specgram must be complex-valued, got dtype {specgram.dtype}")
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask)}")

    if specgram.requires_grad or (mask is not None and mask.requires_grad):
        raise ValueError(
            "psd does not support autograd yet; detach the inputs before calling "
            "to avoid silently losing gradients."
        )

    orig_dtype = specgram.dtype

    # Output dtype follows PyTorch type promotion between the (original) complex
    # spectrogram and the (real) mask, matching torchaudio's ``psd * mask``.
    if mask is not None:
        out_dtype = torch.promote_types(orig_dtype, mask.dtype)
    else:
        out_dtype = orig_dtype

    # comp_dtype is the precision used inside the kernel.  It is at least
    # complex64 (complex32/fp16 accumulation is numerically unstable) and at
    # least as wide as out_dtype (mask-driven promotion).
    comp_dtype = out_dtype
    if comp_dtype == torch.complex32:
        comp_dtype = torch.complex64
    specgram = specgram.to(comp_dtype)

    channel = specgram.shape[-3]
    freq = specgram.shape[-2]
    time = specgram.shape[-1]
    batch_shape = specgram.shape[:-3]
    batch = 1
    for s in batch_shape:
        batch *= s

    if mask is not None:
        if mask.ndim != specgram.ndim - 1:
            raise ValueError(
                f"mask must have {specgram.ndim - 1} dimensions "
                f"(..., freq, time), got {mask.ndim}"
            )
        # After torchaudio's transpose(-3, -2), specgram is (..., freq, channel, time).
        # mask must match (..., freq, time): all dims except the last two of the
        # transposed specgram, plus the time dim.
        expected_freq_time = specgram.shape[:-3] + (specgram.shape[-2],)
        if mask.shape[:-1] != expected_freq_time or mask.shape[-1] != time:
            raise ValueError(
                "The dimensions of mask except the channel dimension should be the "
                f"same as specgram. Found {mask.shape} for mask and "
                f"{specgram.shape} for specgram."
            )
        if mask.device != specgram.device:
            raise ValueError(
                f"mask device {mask.device} does not match specgram device "
                f"{specgram.device}"
            )

    device = specgram.device

    # Degenerate shapes: torchaudio's ``sum`` over an empty time axis yields
    # all-zero PSD, so return zeros (never uninitialized memory).
    out_shape = batch_shape + (freq, channel, channel)
    if freq == 0 or channel == 0 or time == 0:
        return torch.zeros(out_shape, dtype=out_dtype, device=device)

    # Flatten leading dims so the kernel sees (batch, channel, freq, time).
    specgram_flat = specgram.reshape(batch, channel, freq, time)
    spec_real = specgram_flat.real
    spec_imag = specgram_flat.imag

    out = torch.empty(
        (batch, freq, channel, channel), dtype=comp_dtype, device=device
    )
    out_real = out.real
    out_imag = out.imag

    # BLOCK_C must be a power of two >= channel for tl.arange.
    block_c = 1
    while block_c < channel:
        block_c *= 2
    BLOCK_T = 64

    has_mask = mask is not None
    if has_mask:
        # Convert mask to the real precision matching comp_dtype so the
        # kernel's multiply-accumulate lands in comp_dtype (which is at least
        # as wide as the promoted output).  Integer masks are promoted to the
        # matching float width here.
        if comp_dtype == torch.complex128:
            mask_float_dtype = torch.float64
        elif comp_dtype == torch.complex64:
            mask_float_dtype = torch.float32
        else:  # complex32
            mask_float_dtype = torch.float16
        mask_flat = mask.reshape(batch, freq, time).to(mask_float_dtype).contiguous()
        mask_ptr = mask_flat
        mask_stride_b, mask_stride_f, mask_stride_t = mask_flat.stride()
    else:
        mask_ptr = spec_real  # dummy; never accessed when HAS_MASK=False
        mask_stride_b = mask_stride_f = mask_stride_t = 0

    grid = (batch, freq)

    with torch_device_fn.device(device):
        _psd_kernel[grid](
            spec_real,
            spec_imag,
            mask_ptr,
            out_real,
            out_imag,
            channel,
            time,
            spec_real.stride(0),
            spec_real.stride(1),
            spec_real.stride(2),
            spec_real.stride(3),
            mask_stride_b,
            mask_stride_f,
            mask_stride_t,
            out_real.stride(0),
            out_real.stride(1),
            out_real.stride(2),
            out_real.stride(3),
            eps,
            HAS_MASK=has_mask,
            NORMALIZE=normalize,
            BLOCK_C=block_c,
            BLOCK_T=BLOCK_T,
            num_warps=4,
            num_stages=1,
        )

    result = out.reshape(out_shape)
    if result.dtype != out_dtype:
        result = result.to(out_dtype)
    return result
