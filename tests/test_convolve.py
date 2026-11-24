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

import numpy as np
import pytest
import torch

import flag_audio
import torchaudio

from .accuracy_utils import gems_assert_close, to_reference


@pytest.mark.convolve
@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        ((64,), (8,)),            # 1D：短核
        ((400,), (7,)),           # 信号长、核短
        ((400,), (400,)),         # valid 退化成单点
        ((7,), (400,)),           # x 比 y 短，验 same 取原始 x 长度
        ((2, 400), (1, 64)),      # 共享核 + 多 batch（y 前导维为 1）
        ((3, 2, 16), (3, 2, 5)),  # 逐信号配对（各自带核）
        ((2, 1, 16), (1, 3, 5)),  # 真广播
        ((1,), (5,)),             # n_len=1 退化
        ((32,), (1,)),            # m_len=1 退化
        ((128,), (1,)),           # out_len 恰好落在 BLOCK_OUT 边界
        ((129,), (1,)),           # out_len 跨 BLOCK_OUT 边界
        ((1, 200), (1, 65)),      # M 跨 BLOCK_T 边界
        ((1, 200), (1, 64)),      # M 恰好等于 BLOCK_T
        ((3, 2, 16), (1, 1, 5)),  # ndim>=3 的共享核
    ],
)
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy_convolve(x_shape, y_shape, mode, dtype):
    x = torch.randn(x_shape, dtype=dtype, device=flag_audio.device)
    y = torch.randn(y_shape, dtype=dtype, device=flag_audio.device)
    ref_x = to_reference(x, False)
    ref_y = to_reference(y, False)

    ref_out = torchaudio.functional.convolve(ref_x, ref_y, mode)
    res_out = flag_audio.ops.convolve(x, y, mode)

    # 累加顺序与 cuDNN 不同（我们全程 fp32 顺序累加），按 dtype 放宽绝对容差
    atol = {
        torch.float32: 1e-4,
        torch.float16: 5e-3,
        torch.bfloat16: 4e-2,
    }[dtype]
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.convolve
def test_convolve_invalid_shapes():
    x = torch.randn(2, 8, device=flag_audio.device)
    # 两个操作数维度数不一致
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x, torch.randn(8, device=flag_audio.device))
    # 前导维不可广播
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x, torch.randn(3, 8, device=flag_audio.device))
    # 非法 mode
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x, torch.randn(2, 4, device=flag_audio.device), "invalid")


@pytest.mark.convolve
def test_convolve_unsupported_inputs():
    x = torch.randn(2, 8, device=flag_audio.device)
    y = torch.randn(2, 4, device=flag_audio.device)
    # requires_grad：暂不支持 autograd，必须显式报错而非静默丢梯度
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x.clone().requires_grad_(), y)
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x, y.clone().requires_grad_())
    # complex dtype 不支持
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x.to(torch.complex64), y)


@pytest.mark.convolve
def test_convolve_empty_batch():
    # batch=0：直接返回空结果，不启动 0 维 grid（torchaudio 此时也会失败，无法对拍）
    x = torch.randn(0, 8, device=flag_audio.device)
    y = torch.randn(1, 4, device=flag_audio.device)
    out = flag_audio.ops.convolve(x, y, "full")
    assert out.shape == (0, 11)
    assert out.dtype == torch.float32


@pytest.mark.convolve
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_accuracy_convolve_noncontiguous(mode):
    # 覆盖 wrapper 里的 .contiguous() 归一化路径：切片、转置、非连续核
    y = torch.randn(1, 8, dtype=torch.float32, device=flag_audio.device)
    base = torch.randn(2, 800, dtype=torch.float32, device=flag_audio.device)
    x_slice = base[:, ::2]  # stride=(800, 2)
    x_transposed = torch.randn(400, 2, dtype=torch.float32, device=flag_audio.device).t()
    for x in (x_slice, x_transposed):
        assert not x.is_contiguous()
        ref_out = torchaudio.functional.convolve(to_reference(x, False), to_reference(y, False), mode)
        res_out = flag_audio.ops.convolve(x, y, mode)
        gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)

    # 非连续的核
    x = torch.randn(2, 400, dtype=torch.float32, device=flag_audio.device)
    y_nc = torch.randn(1, 800, dtype=torch.float32, device=flag_audio.device)[:, ::2]
    assert not y_nc.is_contiguous()
    gems_assert_close(
        flag_audio.ops.convolve(x, y_nc, mode),
        torchaudio.functional.convolve(to_reference(x, False), to_reference(y_nc, False), mode),
        torch.float32,
        atol=1e-4,
    )


@pytest.mark.convolve
def test_accuracy_convolve_large_batch():
    # 回归：batch 若落在 gridDim.y 会撞上 65535 上限而启动失败
    x = torch.randn(70000, 4, dtype=torch.float32, device=flag_audio.device)
    y = torch.randn(1, 4, dtype=torch.float32, device=flag_audio.device)
    ref_out = torchaudio.functional.convolve(to_reference(x, False), to_reference(y, False), "full")
    res_out = flag_audio.ops.convolve(x, y, "full")
    gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)


@pytest.mark.convolve
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_accuracy_convolve_mode_as_kwarg(mode):
    # benchmark 通过 kwargs 传 mode（字符串无法经 unpack_to_args_kwargs 位置传递）
    x = torch.randn(2, 40, dtype=torch.float32, device=flag_audio.device)
    y = torch.randn(1, 5, dtype=torch.float32, device=flag_audio.device)
    res_out = flag_audio.ops.convolve(x, y, mode=mode)
    ref_out = torchaudio.functional.convolve(to_reference(x, False), to_reference(y, False), mode)
    gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)


@pytest.mark.convolve
def test_accuracy_convolve_large_magnitude():
    # 大幅度输入下 fp32 累加的相对误差（相对输出幅度）应保持在 1e-3 以内，
    # 防止将来误改成低精度累加
    scale = 1e4
    x = torch.randn(1, 400, dtype=torch.float32, device=flag_audio.device) * scale
    y = torch.randn(1, 400, dtype=torch.float32, device=flag_audio.device) * scale
    golden = torch.from_numpy(
        np.convolve(
            x.cpu().double().numpy()[0],
            y.cpu().double().numpy()[0],
            mode="full",
        )
    )
    got = flag_audio.ops.convolve(x, y, "full").cpu().double()[0]
    assert torch.isfinite(golden).all() and golden.abs().max() > 0, "Invalid CPU reference"
    assert torch.isfinite(got).all(), "Convolution output contains non-finite values"
    rel_err = (got - golden).abs().max() / golden.abs().max()
    assert rel_err < 1e-3, f"大幅度下相对误差过大: {rel_err:.3e}"
