import torch
import triton
import triton.language as tl
from typing import Union


def _get_mask_param(mask_param: int, p: float, dim_size: int) -> int:
    if p == 1.0:
        return mask_param
    max_v = min(mask_param, int(dim_size * p))
    return max_v


@triton.jit
def mask_along_axis_iid_kernel(
    specgrams_ptr,
    output_ptr,
    value_ptr,
    min_value_ptr,
    mask_value_ptr,
    dim1,
    dim2,
    is_freq: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total_elements = dim1 * dim2
    mask = offsets < total_elements

    dim1_id = offsets // dim2
    dim2_id = offsets % dim2
    mask_start = tl.load(min_value_ptr + batch_id).to(tl.int64)
    mask_end = mask_start + tl.load(value_ptr + batch_id).to(tl.int64)
    mask_val = tl.load(mask_value_ptr)

    if is_freq:
        in_mask = (dim1_id >= mask_start) & (dim1_id < mask_end)
    else:
        in_mask = (dim2_id >= mask_start) & (dim2_id < mask_end)

    x = tl.load(specgrams_ptr + batch_id * total_elements + offsets, mask=mask)
    y = tl.where(in_mask, mask_val.to(x.dtype), x)
    tl.store(output_ptr + batch_id * total_elements + offsets, y, mask=mask)


def mask_along_axis_iid_triton(
    specgrams: torch.Tensor,
    mask_param: int,
    mask_value: Union[float, torch.Tensor],
    axis: int,
    p: float = 1.0,
) -> torch.Tensor:
    dim = specgrams.dim()
    if dim < 3:
        raise ValueError(
            f"Spectrogram must have at least three dimensions ({dim} given)."
        )
    if axis not in [dim - 2, dim - 1]:
        raise ValueError(
            "Only Frequency and Time masking are supported"
            f" (axis {dim - 2} and axis {dim - 1} supported; {axis} given)."
        )
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    axis_size = specgrams.size(axis)
    effective_mask_param = _get_mask_param(mask_param, p, axis_size)
    if effective_mask_param < 1:
        return specgrams

    original_shape = specgrams.shape
    specgrams_2d = specgrams.reshape(
        -1, original_shape[-2] * original_shape[-1]
    )
    batch_size, inner_dim = specgrams_2d.shape
    output = torch.empty_like(specgrams_2d)

    device = specgrams.device
    dtype = specgrams.dtype

    value = torch.rand(batch_size, device=device, dtype=dtype) * effective_mask_param
    min_value = torch.rand(batch_size, device=device, dtype=dtype) * (axis_size - value)

    if not isinstance(mask_value, torch.Tensor):
        mask_value = torch.tensor([mask_value], device=device, dtype=dtype)

    is_freq = axis == dim - 2
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(inner_dim, BLOCK_SIZE), batch_size)

    mask_along_axis_iid_kernel[grid](
        specgrams_2d,
        output,
        value,
        min_value,
        mask_value,
        original_shape[-2],
        original_shape[-1],
        is_freq=is_freq,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(original_shape)