import torch
import triton
import triton.language as tl


def _get_mask_param(mask_param, p, dim_size):
    if p == 1.0:
        return mask_param
    else:
        return min(mask_param, int(dim_size * p))


@triton.jit
def mask_along_axis_kernel(
    input_ptr,
    output_ptr,
    dim1,
    dim2,
    mask_start,
    mask_end,
    mask_value,
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

    if is_freq:
        in_mask = (dim1_id >= mask_start) & (dim1_id < mask_end)
    else:
        in_mask = (dim2_id >= mask_start) & (dim2_id < mask_end)

    x = tl.load(input_ptr + batch_id * total_elements + offsets, mask=mask)
    y = tl.where(in_mask, mask_value.to(x.dtype), x)
    tl.store(output_ptr + batch_id * total_elements + offsets, y, mask=mask)


def mask_along_axis_triton(
    specgram: torch.Tensor,
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = 1.0,
) -> torch.Tensor:
    dim = specgram.dim()
    if dim < 2:
        raise ValueError(
            f"Spectrogram must have at least two dimensions ({dim} given)."
        )
    if axis not in [dim - 2, dim - 1]:
        raise ValueError(
            "Only Frequency and Time masking are supported "
            f"(axis {dim-2} and {dim-1} supported; {axis} given)."
        )
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be between 0.0 and 1.0 ({p} given).")

    max_v = _get_mask_param(mask_param, p, specgram.shape[axis])
    if max_v < 1:
        return specgram

    device = specgram.device
    v = torch.rand(1, device=device).item() * max_v
    v0 = torch.rand(1, device=device).item() * (specgram.shape[axis] - v)
    mask_start = int(v0)
    mask_end = int(v0) + int(v)

    original_shape = specgram.shape
    specgram_2d = specgram.reshape(
        -1, original_shape[-2] * original_shape[-1]
    )
    batch_size, inner_dim = specgram_2d.shape
    is_freq = axis == dim - 2

    output = torch.empty_like(specgram_2d)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(inner_dim, BLOCK_SIZE), batch_size)

    mask_along_axis_kernel[grid](
        specgram_2d,
        output,
        original_shape[-2],
        original_shape[-1],
        mask_start,
        mask_end,
        mask_value,
        is_freq,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(original_shape)
