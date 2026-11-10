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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


_MAX_CHANNEL = 32
_MAX_ITER = 1
_SQUARE_ITER = 32


@triton.jit
def _square_principal(
    real,
    imag,
    vec_real,
    vec_imag,
    residual,
    residual_tol,
    channel,
    square_iter,
    BLOCK_C: tl.constexpr,
):
    """Find a dominant-magnitude eigenvector by normalized matrix squaring."""
    offs = tl.arange(0, BLOCK_C)
    rows = offs[:, None]
    cols = offs[None, :]
    lane_mask = offs < channel
    square_real = real
    square_imag = imag
    eigenvalue = residual * 0.0
    for _ in range(square_iter):
        if residual > residual_tol:
            square_scale = tl.max(
                tl.sum(
                    tl.sqrt(square_real * square_real + square_imag * square_imag),
                    axis=1,
                ),
                axis=0,
            )
            square_scale = tl.where(square_scale > 0.0, square_scale, 1.0)
            square_real = square_real / square_scale
            square_imag = square_imag / square_scale
            next_real = tl.zeros((BLOCK_C, BLOCK_C), dtype=real.dtype)
            next_imag = tl.zeros((BLOCK_C, BLOCK_C), dtype=real.dtype)
            for k in range(channel):
                col_real = tl.sum(tl.where(cols == k, square_real, 0.0), axis=1)
                col_imag = tl.sum(tl.where(cols == k, square_imag, 0.0), axis=1)
                next_real += (
                    col_real[:, None] * col_real[None, :]
                    + col_imag[:, None] * col_imag[None, :]
                )
                next_imag += (
                    col_imag[:, None] * col_real[None, :]
                    - col_real[:, None] * col_imag[None, :]
                )
            square_real = next_real
            square_imag = next_imag

            square_diagonal = tl.sum(
                tl.where(rows == cols, square_real, 0.0), axis=1
            )
            square_pivot = tl.argmax(
                tl.where(lane_mask, square_diagonal, float("-inf")), axis=0
            )
            vec_real = tl.sum(
                tl.where(cols == square_pivot, square_real, 0.0), axis=1
            )
            vec_imag = tl.sum(
                tl.where(cols == square_pivot, square_imag, 0.0), axis=1
            )
            vec_norm = tl.sqrt(tl.sum(vec_real * vec_real + vec_imag * vec_imag, axis=0))
            vec_real = tl.where(
                vec_norm > 0.0, vec_real / vec_norm, (offs == 0).to(real.dtype)
            )
            vec_imag = tl.where(vec_norm > 0.0, vec_imag / vec_norm, 0.0)
            vec_real = tl.where(lane_mask, vec_real, 0.0)
            vec_imag = tl.where(lane_mask, vec_imag, 0.0)

            prod_real = tl.sum(
                real * vec_real[None, :] - imag * vec_imag[None, :], axis=1
            )
            prod_imag = tl.sum(
                real * vec_imag[None, :] + imag * vec_real[None, :], axis=1
            )
            eigenvalue = tl.sum(vec_real * prod_real + vec_imag * prod_imag, axis=0)
            residual_real = prod_real - eigenvalue * vec_real
            residual_imag = prod_imag - eigenvalue * vec_imag
            residual = tl.sqrt(
                tl.sum(
                    residual_real * residual_real + residual_imag * residual_imag,
                    axis=0,
                )
            )
    return vec_real, vec_imag, residual, eigenvalue


@triton.jit
def _rtf_evd_kernel(
    in_real_ptr,
    in_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    matrix_offsets_ptr,
    channel,
    max_iter,
    square_iter,
    in_stride_m,
    in_stride_r,
    in_stride_c,
    out_stride_m,
    out_stride_c,
    BLOCK_C: tl.constexpr,
    FP64: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
    """Compute the principal eigenvector of one Hermitian matrix per program.

    Only the lower triangle is read, matching ``torch.linalg.eigh(UPLO='L')``.
    Matrix squaring targets the largest-magnitude eigenvalue.  An on-demand
    Gershgorin shift is applied only when that eigenvalue is negative, which
    preserves the largest-algebraic-eigenvalue semantics for general inputs.
    """
    matrix_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    rows = offs[:, None]
    cols = offs[None, :]
    lane_mask = offs < channel
    matrix_mask = lane_mask[:, None] & lane_mask[None, :]
    lower_mask = matrix_mask & (rows >= cols)

    if USE_OFFSETS:
        base = tl.load(matrix_offsets_ptr + matrix_idx)
    else:
        base = matrix_idx * in_stride_m
    addrs = base + rows * in_stride_r + cols * in_stride_c
    real = tl.load(in_real_ptr + addrs, mask=lower_mask, other=0.0)
    imag_lower = tl.load(in_imag_ptr + addrs, mask=lower_mask, other=0.0)
    imag_lower = tl.where(rows > cols, imag_lower, 0.0)

    # Reconstruct the Hermitian matrix from its lower triangle.  The imaginary
    # part of the diagonal is ignored, as in LAPACK Hermitian eigensolvers.
    real = tl.where(rows >= cols, real, tl.trans(real))
    imag = tl.where(rows > cols, imag_lower, -tl.trans(imag_lower))
    real = tl.where(matrix_mask, real, 0.0)
    imag = tl.where(matrix_mask, imag, 0.0)

    diagonal = tl.sum(tl.where(rows == cols, real, 0.0), axis=1)
    magnitude = tl.sqrt(real * real + imag * imag)
    radius = tl.sum(tl.where(rows != cols, magnitude, 0.0), axis=1)
    lower_bound = tl.min(tl.where(lane_mask, diagonal - radius, float("inf")), axis=0)
    shift = tl.maximum(-lower_bound, 0.0)
    magnitude = tl.sqrt(real * real + imag * imag)

    # The strongest diagonal column is exactly the principal direction for a
    # rank-one PSD and is a robust deterministic initial guess in general.
    pivot = tl.argmax(tl.where(lane_mask, diagonal, float("-inf")), axis=0)
    vec_real = tl.sum(tl.where(cols == pivot, real, 0.0), axis=1)
    vec_imag = tl.sum(tl.where(cols == pivot, imag, 0.0), axis=1)
    vec_norm = tl.sqrt(tl.sum(vec_real * vec_real + vec_imag * vec_imag, axis=0))
    vec_real = tl.where(vec_norm > 0.0, vec_real / vec_norm, (offs == 0).to(real.dtype))
    vec_imag = tl.where(vec_norm > 0.0, vec_imag / vec_norm, 0.0)
    vec_real = tl.where(lane_mask, vec_real, 0.0)
    vec_imag = tl.where(lane_mask, vec_imag, 0.0)

    matrix_norm = tl.max(tl.sum(magnitude, axis=1), axis=0)
    if FP64:
        tolerance = 1.0e-12
    else:
        tolerance = 2.0e-6
    residual_tol = tolerance * matrix_norm
    residual = matrix_norm + float("inf")
    eigenvalue = matrix_norm * 0.0

    for _ in range(max_iter):
        prod_real = tl.sum(
            real * vec_real[None, :] - imag * vec_imag[None, :], axis=1
        )
        prod_imag = tl.sum(
            real * vec_imag[None, :] + imag * vec_real[None, :], axis=1
        )
        eigenvalue = tl.sum(
            vec_real * prod_real + vec_imag * prod_imag, axis=0
        )
        residual_real = prod_real - eigenvalue * vec_real
        residual_imag = prod_imag - eigenvalue * vec_imag
        residual = tl.sqrt(
            tl.sum(residual_real * residual_real + residual_imag * residual_imag, axis=0)
        )
        if residual > residual_tol:
            prod_norm = tl.sqrt(
                tl.sum(prod_real * prod_real + prod_imag * prod_imag, axis=0)
            )
            vec_real = tl.where(prod_norm > 0.0, prod_real / prod_norm, vec_real)
            vec_imag = tl.where(prod_norm > 0.0, prod_imag / prod_norm, vec_imag)

    # Matrix squaring finds the eigenvalue of largest magnitude.  If that
    # eigenvalue is negative, repeat with the on-demand Gershgorin shift so
    # the result is instead the largest algebraic eigenvalue.
    if residual <= residual_tol:
        if eigenvalue < 0.0:
            residual = matrix_norm + float("inf")
    if residual > residual_tol:
        vec_real, vec_imag, residual, eigenvalue = _square_principal(
            real,
            imag,
            vec_real,
            vec_imag,
            residual,
            residual_tol,
            channel,
            square_iter,
            BLOCK_C,
        )
        if eigenvalue < 0.0:
            real = tl.where(rows == cols, real + shift, real)
            residual_tol = tolerance * (matrix_norm + shift)
            residual = matrix_norm + float("inf")
            vec_real, vec_imag, residual, eigenvalue = _square_principal(
                real,
                imag,
                vec_real,
                vec_imag,
                residual,
                residual_tol,
                channel,
                square_iter,
                BLOCK_C,
            )

    # Fix the otherwise arbitrary complex phase for reproducible output.  The
    # largest component is rotated onto the positive real axis.
    phase_idx = tl.argmax(vec_real * vec_real + vec_imag * vec_imag, axis=0)
    phase_real = tl.sum(tl.where(offs == phase_idx, vec_real, 0.0), axis=0)
    phase_imag = tl.sum(tl.where(offs == phase_idx, vec_imag, 0.0), axis=0)
    phase_norm = tl.sqrt(phase_real * phase_real + phase_imag * phase_imag)
    out_real = (vec_real * phase_real + vec_imag * phase_imag) / phase_norm
    out_imag = (vec_imag * phase_real - vec_real * phase_imag) / phase_norm

    out_base = matrix_idx * out_stride_m
    tl.store(
        out_real_ptr + out_base + offs * out_stride_c,
        out_real,
        mask=lane_mask,
    )
    tl.store(
        out_imag_ptr + out_base + offs * out_stride_c,
        out_imag,
        mask=lane_mask,
    )


def rtf_evd(psd_s: torch.Tensor) -> torch.Tensor:
    """Estimate the RTF from the principal eigenvector of each speech PSD.

    Args:
        psd_s: Complex Hermitian PSD of shape ``(..., freq, channel, channel)``.
            Only its lower triangle is read.  Non-contiguous inputs use an
            int64 matrix-offset table instead of a full PSD copy.

    Returns:
        A unit-norm complex tensor of shape ``(..., freq, channel)``.  The
        largest component is rotated onto the positive real axis.

    Notes:
        This forward-only implementation supports complex64/complex128 and
        channel sizes from 1 through 32.  Autograd and channels above 32 are
        not supported.  The solver checks the initial column with one matvec
        before switching to at most 32 normalized matrix-squaring steps.
    """
    if not isinstance(psd_s, torch.Tensor):
        raise TypeError(f"psd_s must be a torch.Tensor. Found {type(psd_s).__name__}.")
    if not psd_s.is_complex():
        raise TypeError(
            "The type of psd_s must be ``torch.cfloat`` or ``torch.cdouble``. "
            f"Found {psd_s.dtype}."
        )
    if psd_s.ndim < 2:
        raise ValueError("psd_s must have at least two dimensions.")
    if psd_s.shape[-1] != psd_s.shape[-2]:
        raise ValueError(
            "The last two dimensions of psd_s should be the same. "
            f"Found {psd_s.shape}."
        )
    if psd_s.dtype not in (torch.complex64, torch.complex128):
        raise TypeError(f"rtf_evd does not support {psd_s.dtype}.")
    if psd_s.requires_grad:
        raise RuntimeError("rtf_evd currently does not support autograd.")

    channel = psd_s.shape[-1]
    if channel == 0:
        raise IndexError("index -1 is out of bounds for dimension 1 with size 0")
    if channel > _MAX_CHANNEL:
        raise ValueError(
            f"rtf_evd supports at most {_MAX_CHANNEL} channels. Found {channel}."
        )

    out = torch.empty(
        psd_s.shape[:-1], dtype=psd_s.dtype, device=psd_s.device
    )
    if psd_s.numel() == 0:
        return out

    psd_s = psd_s.resolve_conj().resolve_neg()
    real_view = torch.view_as_real(psd_s)
    try:
        matrices = real_view.view(-1, channel, channel, 2)
        matrix_offsets = real_view
        use_offsets = False
    except RuntimeError:
        matrices = real_view
        use_offsets = True
        matrix_count = out.numel() // channel
        linear_index = torch.arange(matrix_count, device=psd_s.device)
        matrix_offsets = torch.zeros(
            matrix_count, dtype=torch.int64, device=psd_s.device
        )
        for dim in range(psd_s.ndim - 3, -1, -1):
            coordinate = linear_index % psd_s.shape[dim]
            linear_index = linear_index // psd_s.shape[dim]
            matrix_offsets += coordinate * real_view.stride(dim)
    out_view = torch.view_as_real(out).reshape(-1, channel, 2)
    matrix_count = out_view.shape[0]

    with torch_device_fn.device(psd_s.device):
        _rtf_evd_kernel[(matrix_count,)](
            matrices[..., 0],
            matrices[..., 1],
            out_view[..., 0],
            out_view[..., 1],
            matrix_offsets,
            channel,
            _MAX_ITER,
            _SQUARE_ITER,
            0 if use_offsets else matrices.stride(0),
            matrices.stride(-3),
            matrices.stride(-2),
            out_view.stride(0),
            out_view.stride(1),
            BLOCK_C=max(1, triton.next_power_of_2(channel)),
            FP64=psd_s.dtype == torch.complex128,
            USE_OFFSETS=use_offsets,
            num_warps=4,
        )
    return out
