import torch
import triton
import triton.language as tl


@triton.jit
def preemphasis_kernel(
    x_ptr,
    y_ptr,
    coeff: float,
    x_stride_0: tl.constexpr,
    x_stride_1: tl.constexpr,
    y_stride_0: tl.constexpr,
    y_stride_1: tl.constexpr,
    last_dim_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = tl.program_id(1)

    logic_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = logic_offsets < last_dim_len
    x = tl.load(x_ptr + row_id * x_stride_0 + logic_offsets * x_stride_1, mask=mask)

    pos = logic_offsets % last_dim_len
    prev_mask = (logic_offsets >= 1) & mask & (pos != 0)
    prev_offsets = (logic_offsets - 1) * x_stride_1

    prev_x = tl.load(
        x_ptr + row_id * x_stride_0 + prev_offsets, mask=prev_mask, other=0.0
    )
    result = tl.where(pos == 0, x, x - coeff * prev_x)
    tl.store(
        y_ptr + row_id * y_stride_0 + logic_offsets * y_stride_1, result, mask=mask
    )


def preemphasis_triton(waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    waveform_out = torch.empty_like(waveform).reshape(-1, waveform.shape[-1])

    last_dim_len = waveform.shape[-1]
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(last_dim_len, BLOCK_SIZE), waveform_out.size(1))

    preemphasis_kernel[grid](
        waveform,
        waveform_out,
        coeff,
        waveform.stride(0),
        waveform.stride(1),
        waveform_out.stride(0),
        waveform_out.stride(1),
        last_dim_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return waveform_out.reshape(waveform.shape)