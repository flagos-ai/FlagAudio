import torch
import triton
import triton.language as tl


@triton.jit
def DB_to_amplitude_triton_kernel(
    x_ptr,
    output_ptr,
    ref: float,
    power: float,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    log10 = 2.302585092994046
    x = tl.load(x_ptr + offsets, mask=mask)
    result = ref * tl.exp(0.1 * x * log10 * power)

    tl.store(output_ptr + offsets, result, mask=mask)


def DB_to_amplitude_triton(x: torch.Tensor, ref: float, power: float) -> torch.Tensor:

    BLOCK_SIZE = 1024
    n_elements = x.numel()
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)

    output = torch.empty_like(x)

    DB_to_amplitude_triton_kernel[grid_size](
        x, output, ref, power, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    return output