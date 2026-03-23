import torch
import triton
import triton.language as tl


@triton.jit
def gain_kernel(
    input_ptr,
    output_ptr,
    gain_db: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    ratio = 10 ** (gain_db / 20)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    output_vals = input_vals * ratio

    tl.store(output_ptr + offsets, output_vals, mask=mask)


def gain_triton(input_tensor: torch.Tensor, gain_db: float = 1.0) -> torch.Tensor:
    if gain_db == 1.0:
        return input_tensor

    output_tensor = torch.empty_like(input_tensor)

    BLOCK_SIZE = 1024
    n_elements = input_tensor.numel()
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)

    gain_kernel[grid_size](
        input_tensor,
        output_tensor,
        gain_db=gain_db,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_tensor