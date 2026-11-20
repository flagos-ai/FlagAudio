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


_POWER_ITER = 1
_SQUARE_ITER = 64
_LARGE_TILE = 32


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

    # Eigenvectors are invariant under a positive matrix scale.  Normalize
    # before forming squares or norms so extreme finite inputs do not overflow
    # or underflow merely because of their units.
    component_scale = tl.max(
        tl.max(tl.maximum(tl.abs(real), tl.abs(imag)), axis=1), axis=0
    )
    component_scale = tl.where(component_scale > 0.0, component_scale, 1.0)
    real = real / component_scale
    imag = imag / component_scale

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
        if residual > residual_tol:
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
                tl.sum(
                    residual_real * residual_real + residual_imag * residual_imag,
                    axis=0,
                )
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

    # Never silently return an unconverged candidate as a plausible RTF.  NaN
    # keeps the native return signature while making failure visible without a
    # device-to-host synchronization on every call.
    converged = residual <= residual_tol
    vec_real = tl.where(converged, vec_real, float("nan"))
    vec_imag = tl.where(converged, vec_imag, float("nan"))

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


@triton.jit
def _load_hermitian_tile(
    in_real_ptr,
    in_imag_ptr,
    base,
    in_stride_r,
    in_stride_c,
    channel,
    scale,
    shift,
    rows,
    cols,
):
    """Load a scaled Hermitian tile while reading only the input lower triangle."""
    row = rows[:, None]
    col = cols[None, :]
    valid = (row < channel) & (col < channel)
    lower = valid & (row >= col)
    upper = valid & (row < col)
    real_lower = tl.load(
        in_real_ptr + base + row * in_stride_r + col * in_stride_c,
        mask=lower,
        other=0.0,
    )
    real_upper = tl.load(
        in_real_ptr + base + col * in_stride_r + row * in_stride_c,
        mask=upper,
        other=0.0,
    )
    imag_lower = tl.load(
        in_imag_ptr + base + row * in_stride_r + col * in_stride_c,
        mask=lower & (row > col),
        other=0.0,
    )
    imag_upper = tl.load(
        in_imag_ptr + base + col * in_stride_r + row * in_stride_c,
        mask=upper,
        other=0.0,
    )
    real = (real_lower + real_upper) / scale
    real = tl.where((row == col) & valid, real + shift, real)
    imag = (imag_lower - imag_upper) / scale
    return real, imag


@triton.jit
def _large_component_scale(
    in_real_ptr,
    in_imag_ptr,
    base,
    in_stride_r,
    in_stride_c,
    channel,
    BLOCK_T: tl.constexpr,
):
    scale = tl.sum(
        tl.zeros((1,), dtype=in_real_ptr.dtype.element_ty), axis=0
    )
    for row_start in range(0, channel, BLOCK_T):
        rows = row_start + tl.arange(0, BLOCK_T)
        for col_start in range(0, channel, BLOCK_T):
            cols = col_start + tl.arange(0, BLOCK_T)
            row = rows[:, None]
            col = cols[None, :]
            lower = (row < channel) & (col < channel) & (row >= col)
            real = tl.load(
                in_real_ptr + base + row * in_stride_r + col * in_stride_c,
                mask=lower,
                other=0.0,
            )
            imag = tl.load(
                in_imag_ptr + base + row * in_stride_r + col * in_stride_c,
                mask=lower & (row > col),
                other=0.0,
            )
            tile_scale = tl.max(
                tl.max(tl.maximum(tl.abs(real), tl.abs(imag)), axis=1), axis=0
            )
            scale = tl.maximum(scale, tile_scale)
    return tl.where(scale > 0.0, scale, 1.0)


@triton.jit
def _large_norm_and_shift(
    in_real_ptr,
    in_imag_ptr,
    base,
    in_stride_r,
    in_stride_c,
    channel,
    scale,
    BLOCK_T: tl.constexpr,
):
    matrix_norm = tl.sum(
        tl.zeros((1,), dtype=in_real_ptr.dtype.element_ty), axis=0
    )
    lower_bound = matrix_norm + float("inf")
    for row_start in range(0, channel, BLOCK_T):
        rows = row_start + tl.arange(0, BLOCK_T)
        row_norm = tl.zeros((BLOCK_T,), dtype=in_real_ptr.dtype.element_ty)
        diagonal = tl.zeros((BLOCK_T,), dtype=in_real_ptr.dtype.element_ty)
        radius = tl.zeros((BLOCK_T,), dtype=in_real_ptr.dtype.element_ty)
        for col_start in range(0, channel, BLOCK_T):
            cols = col_start + tl.arange(0, BLOCK_T)
            real, imag = _load_hermitian_tile(
                in_real_ptr,
                in_imag_ptr,
                base,
                in_stride_r,
                in_stride_c,
                channel,
                scale,
                0.0,
                rows,
                cols,
            )
            row = rows[:, None]
            col = cols[None, :]
            magnitude = tl.sqrt(real * real + imag * imag)
            row_norm += tl.sum(magnitude, axis=1)
            diagonal += tl.sum(tl.where(row == col, real, 0.0), axis=1)
            radius += tl.sum(tl.where(row != col, magnitude, 0.0), axis=1)
        row_mask = rows < channel
        matrix_norm = tl.maximum(
            matrix_norm, tl.max(tl.where(row_mask, row_norm, 0.0), axis=0)
        )
        tile_lower = tl.min(
            tl.where(row_mask, diagonal - radius, float("inf")), axis=0
        )
        lower_bound = tl.minimum(lower_bound, tile_lower)
    shift = tl.maximum(-lower_bound, 0.0)
    return matrix_norm, shift


@triton.jit
def _large_matvec(
    in_real_ptr,
    in_imag_ptr,
    vec_real_ptr,
    vec_imag_ptr,
    prod_real_ptr,
    prod_imag_ptr,
    base,
    in_stride_r,
    in_stride_c,
    channel,
    scale,
    shift,
    BLOCK_T: tl.constexpr,
):
    for row_start in range(0, channel, BLOCK_T):
        rows = row_start + tl.arange(0, BLOCK_T)
        row_mask = rows < channel
        prod_real = tl.zeros((BLOCK_T,), dtype=in_real_ptr.dtype.element_ty)
        prod_imag = tl.zeros((BLOCK_T,), dtype=in_real_ptr.dtype.element_ty)
        for col_start in range(0, channel, BLOCK_T):
            cols = col_start + tl.arange(0, BLOCK_T)
            col_mask = cols < channel
            real, imag = _load_hermitian_tile(
                in_real_ptr,
                in_imag_ptr,
                base,
                in_stride_r,
                in_stride_c,
                channel,
                scale,
                shift,
                rows,
                cols,
            )
            vec_real = tl.load(vec_real_ptr + cols, mask=col_mask, other=0.0)
            vec_imag = tl.load(vec_imag_ptr + cols, mask=col_mask, other=0.0)
            prod_real += tl.sum(
                real * vec_real[None, :] - imag * vec_imag[None, :], axis=1
            )
            prod_imag += tl.sum(
                real * vec_imag[None, :] + imag * vec_real[None, :], axis=1
            )
        tl.store(prod_real_ptr + rows, prod_real, mask=row_mask)
        tl.store(prod_imag_ptr + rows, prod_imag, mask=row_mask)


@triton.jit
def _large_residual(
    in_real_ptr,
    in_imag_ptr,
    vec_real_ptr,
    vec_imag_ptr,
    prod_real_ptr,
    prod_imag_ptr,
    base,
    in_stride_r,
    in_stride_c,
    channel,
    scale,
    shift,
    vec_real,
    vec_imag,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_C)
    lane_mask = offs < channel
    tl.store(vec_real_ptr + offs, vec_real, mask=lane_mask)
    tl.store(vec_imag_ptr + offs, vec_imag, mask=lane_mask)
    tl.debug_barrier()
    _large_matvec(
        in_real_ptr,
        in_imag_ptr,
        vec_real_ptr,
        vec_imag_ptr,
        prod_real_ptr,
        prod_imag_ptr,
        base,
        in_stride_r,
        in_stride_c,
        channel,
        scale,
        shift,
        BLOCK_T,
    )
    tl.debug_barrier()
    prod_real = tl.load(prod_real_ptr + offs, mask=lane_mask, other=0.0)
    prod_imag = tl.load(prod_imag_ptr + offs, mask=lane_mask, other=0.0)
    eigenvalue = tl.sum(vec_real * prod_real + vec_imag * prod_imag, axis=0)
    residual_real = prod_real - eigenvalue * vec_real
    residual_imag = prod_imag - eigenvalue * vec_imag
    residual = tl.sqrt(
        tl.sum(residual_real * residual_real + residual_imag * residual_imag, axis=0)
    )
    return prod_real, prod_imag, residual, eigenvalue


@triton.jit
def _large_square_principal(
    in_real_ptr,
    in_imag_ptr,
    real0_ptr,
    imag0_ptr,
    real1_ptr,
    imag1_ptr,
    vec_real_ptr,
    vec_imag_ptr,
    prod_real_ptr,
    prod_imag_ptr,
    base,
    in_stride_r,
    in_stride_c,
    channel,
    scale,
    shift,
    vec_real,
    vec_imag,
    residual,
    residual_tol,
    square_iter,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Tiled normalized squaring with two global-memory matrix buffers."""
    for row_start in range(0, channel, BLOCK_T):
        rows = row_start + tl.arange(0, BLOCK_T)
        for col_start in range(0, channel, BLOCK_T):
            cols = col_start + tl.arange(0, BLOCK_T)
            real, imag = _load_hermitian_tile(
                in_real_ptr,
                in_imag_ptr,
                base,
                in_stride_r,
                in_stride_c,
                channel,
                scale,
                shift,
                rows,
                cols,
            )
            offsets = rows[:, None] * channel + cols[None, :]
            valid = (rows[:, None] < channel) & (cols[None, :] < channel)
            tl.store(real0_ptr + offsets, real, mask=valid)
            tl.store(imag0_ptr + offsets, imag, mask=valid)
    tl.debug_barrier()

    src_real = real0_ptr
    src_imag = imag0_ptr
    dst_real = real1_ptr
    dst_imag = imag1_ptr
    eigenvalue = scale * 0.0
    offs = tl.arange(0, BLOCK_C)
    lane_mask = offs < channel
    for _ in range(square_iter):
        if residual > residual_tol:
            square_scale = scale * 0.0
            for row_start in range(0, channel, BLOCK_T):
                rows = row_start + tl.arange(0, BLOCK_T)
                for col_start in range(0, channel, BLOCK_T):
                    cols = col_start + tl.arange(0, BLOCK_T)
                    offsets = rows[:, None] * channel + cols[None, :]
                    valid = (rows[:, None] < channel) & (cols[None, :] < channel)
                    real = tl.load(src_real + offsets, mask=valid, other=0.0)
                    imag = tl.load(src_imag + offsets, mask=valid, other=0.0)
                    tile_scale = tl.max(
                        tl.max(tl.maximum(tl.abs(real), tl.abs(imag)), axis=1),
                        axis=0,
                    )
                    square_scale = tl.maximum(square_scale, tile_scale)
            square_scale = tl.where(square_scale > 0.0, square_scale, 1.0)

            for row_start in range(0, channel, BLOCK_T):
                rows = row_start + tl.arange(0, BLOCK_T)
                for col_start in range(0, channel, BLOCK_T):
                    cols = col_start + tl.arange(0, BLOCK_T)
                    next_real = tl.zeros(
                        (BLOCK_T, BLOCK_T), dtype=in_real_ptr.dtype.element_ty
                    )
                    next_imag = tl.zeros(
                        (BLOCK_T, BLOCK_T), dtype=in_real_ptr.dtype.element_ty
                    )
                    for k_start in range(0, channel, BLOCK_T):
                        inner = k_start + tl.arange(0, BLOCK_T)
                        left_offsets = rows[:, None] * channel + inner[None, :]
                        right_offsets = inner[:, None] * channel + cols[None, :]
                        left_mask = (rows[:, None] < channel) & (
                            inner[None, :] < channel
                        )
                        right_mask = (inner[:, None] < channel) & (
                            cols[None, :] < channel
                        )
                        left_real = tl.load(
                            src_real + left_offsets, mask=left_mask, other=0.0
                        ) / square_scale
                        left_imag = tl.load(
                            src_imag + left_offsets, mask=left_mask, other=0.0
                        ) / square_scale
                        right_real = tl.load(
                            src_real + right_offsets, mask=right_mask, other=0.0
                        ) / square_scale
                        right_imag = tl.load(
                            src_imag + right_offsets, mask=right_mask, other=0.0
                        ) / square_scale
                        next_real += tl.dot(
                            left_real, right_real, input_precision="ieee"
                        ) - tl.dot(left_imag, right_imag, input_precision="ieee")
                        next_imag += tl.dot(
                            left_real, right_imag, input_precision="ieee"
                        ) + tl.dot(left_imag, right_real, input_precision="ieee")
                    offsets = rows[:, None] * channel + cols[None, :]
                    valid = (rows[:, None] < channel) & (cols[None, :] < channel)
                    tl.store(dst_real + offsets, next_real, mask=valid)
                    tl.store(dst_imag + offsets, next_imag, mask=valid)
            tl.debug_barrier()

            best_diagonal = scale * 0.0 - float("inf")
            pivot = 0
            for start in range(0, channel, BLOCK_T):
                indices = start + tl.arange(0, BLOCK_T)
                mask = indices < channel
                diagonal = tl.load(
                    dst_real + indices * channel + indices, mask=mask, other=0.0
                )
                tile_max = tl.max(
                    tl.where(mask, diagonal, float("-inf")), axis=0
                )
                tile_pivot = start + tl.argmax(
                    tl.where(mask, diagonal, float("-inf")), axis=0
                )
                if tile_max > best_diagonal:
                    best_diagonal = tile_max
                    pivot = tile_pivot
            vec_real = tl.load(
                dst_real + offs * channel + pivot, mask=lane_mask, other=0.0
            )
            vec_imag = tl.load(
                dst_imag + offs * channel + pivot, mask=lane_mask, other=0.0
            )
            vec_norm = tl.sqrt(
                tl.sum(vec_real * vec_real + vec_imag * vec_imag, axis=0)
            )
            vec_real = tl.where(
                vec_norm > 0.0, vec_real / vec_norm, (offs == 0).to(vec_real.dtype)
            )
            vec_imag = tl.where(vec_norm > 0.0, vec_imag / vec_norm, 0.0)
            vec_real = tl.where(lane_mask, vec_real, 0.0)
            vec_imag = tl.where(lane_mask, vec_imag, 0.0)
            prod_real, prod_imag, residual, eigenvalue = _large_residual(
                in_real_ptr,
                in_imag_ptr,
                vec_real_ptr,
                vec_imag_ptr,
                prod_real_ptr,
                prod_imag_ptr,
                base,
                in_stride_r,
                in_stride_c,
                channel,
                scale,
                shift,
                vec_real,
                vec_imag,
                BLOCK_C,
                BLOCK_T,
            )
            tmp_real = src_real
            tmp_imag = src_imag
            src_real = dst_real
            src_imag = dst_imag
            dst_real = tmp_real
            dst_imag = tmp_imag
    return vec_real, vec_imag, residual, eigenvalue


@triton.jit
def _rtf_evd_large_kernel(
    in_real_ptr,
    in_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    matrix_offsets_ptr,
    real0_ptr,
    imag0_ptr,
    real1_ptr,
    imag1_ptr,
    vector_work_ptr,
    channel,
    vector_stride,
    max_iter,
    square_iter,
    in_stride_m,
    in_stride_r,
    in_stride_c,
    out_stride_m,
    out_stride_c,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
    FP64: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
    """Scalable forward solver; no full channel-by-channel register matrix."""
    matrix_idx = tl.program_id(0)
    if USE_OFFSETS:
        base = tl.load(matrix_offsets_ptr + matrix_idx)
    else:
        base = matrix_idx * in_stride_m
    scale = _large_component_scale(
        in_real_ptr,
        in_imag_ptr,
        base,
        in_stride_r,
        in_stride_c,
        channel,
        BLOCK_T,
    )
    matrix_norm, shift = _large_norm_and_shift(
        in_real_ptr,
        in_imag_ptr,
        base,
        in_stride_r,
        in_stride_c,
        channel,
        scale,
        BLOCK_T,
    )

    best_diagonal = scale * 0.0 - float("inf")
    pivot = 0
    for start in range(0, channel, BLOCK_T):
        indices = start + tl.arange(0, BLOCK_T)
        mask = indices < channel
        diagonal = tl.load(
            in_real_ptr + base + indices * in_stride_r + indices * in_stride_c,
            mask=mask,
            other=0.0,
        ) / scale
        tile_max = tl.max(tl.where(mask, diagonal, float("-inf")), axis=0)
        tile_pivot = start + tl.argmax(
            tl.where(mask, diagonal, float("-inf")), axis=0
        )
        if tile_max > best_diagonal:
            best_diagonal = tile_max
            pivot = tile_pivot

    offs = tl.arange(0, BLOCK_C)
    lane_mask = offs < channel
    lower = lane_mask & (offs >= pivot)
    upper = lane_mask & (offs < pivot)
    vec_real = tl.load(
        in_real_ptr + base + offs * in_stride_r + pivot * in_stride_c,
        mask=lower,
        other=0.0,
    ) + tl.load(
        in_real_ptr + base + pivot * in_stride_r + offs * in_stride_c,
        mask=upper,
        other=0.0,
    )
    vec_imag = tl.load(
        in_imag_ptr + base + offs * in_stride_r + pivot * in_stride_c,
        mask=lower & (offs > pivot),
        other=0.0,
    ) - tl.load(
        in_imag_ptr + base + pivot * in_stride_r + offs * in_stride_c,
        mask=upper,
        other=0.0,
    )
    vec_real = vec_real / scale
    vec_imag = vec_imag / scale
    vec_norm = tl.sqrt(tl.sum(vec_real * vec_real + vec_imag * vec_imag, axis=0))
    vec_real = tl.where(vec_norm > 0.0, vec_real / vec_norm, (offs == 0).to(vec_real.dtype))
    vec_imag = tl.where(vec_norm > 0.0, vec_imag / vec_norm, 0.0)
    vec_real = tl.where(lane_mask, vec_real, 0.0)
    vec_imag = tl.where(lane_mask, vec_imag, 0.0)

    if FP64:
        tolerance = 1.0e-12
    else:
        tolerance = 2.0e-6
    residual_tol = tolerance * matrix_norm
    residual = matrix_norm + float("inf")
    eigenvalue = matrix_norm * 0.0
    vector_offset = matrix_idx.to(tl.int64) * channel
    vec_real_ptr = vector_work_ptr + vector_offset
    vec_imag_ptr = vector_work_ptr + vector_stride + vector_offset
    prod_real_ptr = vector_work_ptr + 2 * vector_stride + vector_offset
    prod_imag_ptr = vector_work_ptr + 3 * vector_stride + vector_offset

    for _ in range(max_iter):
        if residual > residual_tol:
            prod_real, prod_imag, residual, eigenvalue = _large_residual(
                in_real_ptr,
                in_imag_ptr,
                vec_real_ptr,
                vec_imag_ptr,
                prod_real_ptr,
                prod_imag_ptr,
                base,
                in_stride_r,
                in_stride_c,
                channel,
                scale,
                0.0,
                vec_real,
                vec_imag,
                BLOCK_C,
                BLOCK_T,
            )
            if residual > residual_tol:
                prod_norm = tl.sqrt(
                    tl.sum(prod_real * prod_real + prod_imag * prod_imag, axis=0)
                )
                vec_real = tl.where(prod_norm > 0.0, prod_real / prod_norm, vec_real)
                vec_imag = tl.where(prod_norm > 0.0, prod_imag / prod_norm, vec_imag)

    if residual <= residual_tol:
        if eigenvalue < 0.0:
            residual = matrix_norm + float("inf")
    scratch_offset = matrix_idx.to(tl.int64) * channel * channel
    if residual > residual_tol:
        vec_real, vec_imag, residual, eigenvalue = _large_square_principal(
            in_real_ptr,
            in_imag_ptr,
            real0_ptr + scratch_offset,
            imag0_ptr + scratch_offset,
            real1_ptr + scratch_offset,
            imag1_ptr + scratch_offset,
            vec_real_ptr,
            vec_imag_ptr,
            prod_real_ptr,
            prod_imag_ptr,
            base,
            in_stride_r,
            in_stride_c,
            channel,
            scale,
            0.0,
            vec_real,
            vec_imag,
            residual,
            residual_tol,
            square_iter,
            BLOCK_C,
            BLOCK_T,
        )
        if eigenvalue < 0.0:
            residual_tol = tolerance * (matrix_norm + shift)
            residual = matrix_norm + float("inf")
            vec_real, vec_imag, residual, eigenvalue = _large_square_principal(
                in_real_ptr,
                in_imag_ptr,
                real0_ptr + scratch_offset,
                imag0_ptr + scratch_offset,
                real1_ptr + scratch_offset,
                imag1_ptr + scratch_offset,
                vec_real_ptr,
                vec_imag_ptr,
                prod_real_ptr,
                prod_imag_ptr,
                base,
                in_stride_r,
                in_stride_c,
                channel,
                scale,
                shift,
                vec_real,
                vec_imag,
                residual,
                residual_tol,
                square_iter,
                BLOCK_C,
                BLOCK_T,
            )

    converged = residual <= residual_tol
    vec_real = tl.where(converged, vec_real, float("nan"))
    vec_imag = tl.where(converged, vec_imag, float("nan"))
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
        positive channel sizes.  Channels from 32 upward use tiled global-memory
        matrix products instead of a full register matrix.  Autograd is not
        supported.  For finite inputs, a result containing NaN indicates that
        the configured iteration limit was reached before residual convergence.
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
        if channel < 32:
            power_iter = _POWER_ITER
            _rtf_evd_kernel[(matrix_count,)](
                matrices[..., 0],
                matrices[..., 1],
                out_view[..., 0],
                out_view[..., 1],
                matrix_offsets,
                channel,
                power_iter,
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
        else:
            real_dtype = (
                torch.float64 if psd_s.dtype == torch.complex128 else torch.float32
            )
            workspace = torch.empty(
                (2, 2, matrix_count, channel, channel),
                dtype=real_dtype,
                device=psd_s.device,
            )
            vector_work = torch.empty(
                (4, matrix_count, channel), dtype=real_dtype, device=psd_s.device
            )
            _rtf_evd_large_kernel[(matrix_count,)](
                matrices[..., 0],
                matrices[..., 1],
                out_view[..., 0],
                out_view[..., 1],
                matrix_offsets,
                workspace[0, 0],
                workspace[0, 1],
                workspace[1, 0],
                workspace[1, 1],
                vector_work,
                channel,
                matrix_count * channel,
                _POWER_ITER,
                _SQUARE_ITER,
                0 if use_offsets else matrices.stride(0),
                matrices.stride(-3),
                matrices.stride(-2),
                out_view.stride(0),
                out_view.stride(1),
                BLOCK_C=triton.next_power_of_2(channel),
                BLOCK_T=_LARGE_TILE,
                FP64=psd_s.dtype == torch.complex128,
                USE_OFFSETS=use_offsets,
                num_warps=4,
            )
    return out
