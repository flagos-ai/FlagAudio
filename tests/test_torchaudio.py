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

import pytest
import torch
import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas

import flag_audio
import torchaudio

from .accuracy_utils import ASUM_SHAPES, SCALARS, gems_assert_close, to_reference

@pytest.mark.gain
@pytest.mark.parametrize("shape", [(3)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("gain_db", [0.0, 1.0, -6.0, 6.0])
def test_accuracy_gain(shape, gain_db, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.gain(ref_inp1_tensor, gain_db)

    res_out = flag_audio.ops.gain(input_tensor, gain_db)

    gems_assert_close(res_out, ref_out, dtype)



@pytest.mark.DB_to_amplitude
@pytest.mark.parametrize("shape", [(3)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("ref", [3.0])
@pytest.mark.parametrize("power", [0.5])
def test_accuracy_DB_to_amplitude(shape, dtype, ref, power):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.DB_to_amplitude(ref_inp1_tensor, ref, power)
    res_out = flag_audio.ops.DB_to_amplitude(input_tensor, ref, power)

    gems_assert_close(res_out, ref_out, dtype)


def mock_rand(size, **kwargs):
    return torch.ones(size, **kwargs) * 0.6
@pytest.mark.mask_along_axis_iid
@pytest.mark.parametrize("shape", [(2,2,9,4)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("mask_param", [3])
@pytest.mark.parametrize("mask_value", [0.0])
@pytest.mark.parametrize("axis", [3])
@pytest.mark.parametrize("p", [1])
def test_accuracy_mask_along_axis_iid(shape, dtype,mask_param, mask_value, axis, p, monkeypatch):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    monkeypatch.setattr(torch, "rand", mock_rand)
    ref_out = torchaudio.functional.mask_along_axis_iid(ref_inp1_tensor, mask_param, mask_value, axis, p)
    res_out = flag_audio.ops.mask_along_axis_iid(input_tensor, mask_param, mask_value, axis, p)

    gems_assert_close(res_out, ref_out, dtype)

@pytest.mark.mask_along_axis
@pytest.mark.parametrize("shape", [(2,2,4,4)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("mask_param", [6])
@pytest.mark.parametrize("mask_value", [0.0])
@pytest.mark.parametrize("axis", [2])
@pytest.mark.parametrize("p", [0.5])
def test_accuracy_mask_along_axis(shape, dtype,mask_param, mask_value, axis, p, monkeypatch):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    monkeypatch.setattr(torch, "rand", mock_rand)
    ref_out = torchaudio.functional.mask_along_axis(ref_inp1_tensor, mask_param, mask_value, axis, p)
    res_out = flag_audio.ops.mask_along_axis(input_tensor, mask_param, mask_value, axis, p)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.preemphasis
@pytest.mark.parametrize("shape", [(2, 2)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("coeff", [0.97])
def test_accuracy_preemphasis(shape, coeff, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.preemphasis(ref_inp1_tensor, coeff)

    res_out = flag_audio.ops.preemphasis(input_tensor, coeff)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mu_law_encoding
@pytest.mark.parametrize("shape", [(3)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("quantization_channels", [256])
def test_accuracy_mu_law_encoding(shape, dtype, quantization_channels):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.mu_law_encoding(ref_inp1_tensor, quantization_channels)

    res_out = flag_audio.ops.mu_law_encoding(input_tensor, quantization_channels)

    gems_assert_close(res_out, ref_out, torch.int64)


@pytest.mark.amplitude_to_DB
@pytest.mark.parametrize("shape", [(5, 2, 2, 2)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("multiplier", [3.0])
@pytest.mark.parametrize("amin", [0.5])
@pytest.mark.parametrize("db_multiplier", [1.0])
@pytest.mark.parametrize("top_db", [None, 80.0])
def test_accuracy_amplitude_to_DB(shape, dtype, multiplier, amin, db_multiplier, top_db):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.amplitude_to_DB(ref_inp1_tensor, multiplier, amin, db_multiplier, top_db=top_db)
    res_out = flag_audio.ops.amplitude_to_DB(input_tensor, multiplier, amin, db_multiplier, top_db=top_db)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.dcshift
@pytest.mark.parametrize("shape", [(2, 2)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("shift", [0.5])
@pytest.mark.parametrize("limiter_gain", [None, 0.5])
def test_accuracy_dcshift(shape, dtype, shift, limiter_gain):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.dcshift(ref_inp1_tensor, shift, limiter_gain)
    res_out = flag_audio.ops.dcshift(input_tensor, shift, limiter_gain)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.spectral_centroid
@pytest.mark.parametrize("shape", [(1000,), (2, 1000), (2, 2, 1000)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("sample_rate", [16000])
@pytest.mark.parametrize("pad", [0, 1])
@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("hop_length", [160])
@pytest.mark.parametrize("win_length", [400])
def test_accuracy_spectral_centroid(shape, dtype, sample_rate, pad, n_fft, hop_length, win_length):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    windows = torch.hann_window(win_length).to(flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, False)
    ref_win_tensor = to_reference(windows, False)

    ref_out = torchaudio.functional.spectral_centroid(ref_inp1_tensor, sample_rate, pad, ref_win_tensor, n_fft, hop_length, win_length)
    res_out = flag_audio.ops.spectral_centroid(input_tensor, sample_rate, pad, ref_win_tensor, n_fft, hop_length, win_length)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add_noise
@pytest.mark.parametrize(
    "shape",
    [
        (8,),          # 1D：单条 + 标量 snr
        (2, 8),        # 2D：多条
        (256,),        # L 恰好 = BLOCK_SIZE，边界
        (257,),        # L = BLOCK_SIZE+1，第二段只有 1 个点
        (3, 2, 1000),  # 多维 + 非整除多段
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length_ratio", [None, 1.0, 0.5])
def test_accuracy_add_noise(shape, dtype, length_ratio):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    noise = torch.rand(shape, dtype=dtype, device=flag_audio.device)
    snr = torch.rand(shape[:-1], dtype=dtype, device=flag_audio.device)
    length = (
        torch.ones_like(snr, dtype=torch.int64, device=flag_audio.device)
        * int(shape[-1] * length_ratio)
        if length_ratio is not None
        else None
    )
    ref_inp1_tensor = to_reference(input_tensor, False)
    ref_noise_tensor = to_reference(noise, False)
    ref_snr_tensor = to_reference(snr, False)
    ref_length_tensor = to_reference(length, False) if length_ratio is not None else None

    ref_out = torchaudio.functional.add_noise(ref_inp1_tensor, ref_noise_tensor, ref_snr_tensor, lengths=ref_length_tensor)
    res_out = flag_audio.ops.add_noise(ref_inp1_tensor, ref_noise_tensor, ref_snr_tensor, lengths=ref_length_tensor)

    # half 精度下 scale 的舍入路径与 torchaudio 不同（我们全程 fp32），
    # 差异实测仅 1~3 ULP，按 dtype 放宽绝对容差
    atol = {
        torch.float32: 1e-4,
        torch.float16: 5e-3,
        torch.bfloat16: 4e-2,
    }[dtype]
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.add_noise
def test_add_noise_invalid_shapes():
    w = torch.randn(2, 8, device=flag_audio.device)
    n = torch.rand(2, 8, device=flag_audio.device)
    # snr 维度数不匹配：应为 (2,) 而不是标量
    with pytest.raises(ValueError):
        flag_audio.ops.add_noise(w, n, torch.tensor(0.5, device=flag_audio.device))
    # 最后一维长度不一致
    with pytest.raises(ValueError):
        flag_audio.ops.add_noise(
            w,
            torch.rand(2, 7, device=flag_audio.device),
            torch.rand(2, device=flag_audio.device),
        )


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
    res_out = flag_audio.ops.convolve(ref_x, ref_y, mode)

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
        res_out = flag_audio.ops.convolve(to_reference(x, False), to_reference(y, False), mode)
        gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)

    # 非连续的核
    x = torch.randn(2, 400, dtype=torch.float32, device=flag_audio.device)
    y_nc = torch.randn(1, 800, dtype=torch.float32, device=flag_audio.device)[:, ::2]
    assert not y_nc.is_contiguous()
    gems_assert_close(
        flag_audio.ops.convolve(x, y_nc, mode),
        torchaudio.functional.convolve(x, y_nc, mode),
        torch.float32,
        atol=1e-4,
    )


@pytest.mark.convolve
def test_accuracy_convolve_large_batch():
    # 回归：batch 若落在 gridDim.y 会撞上 65535 上限而启动失败
    x = torch.randn(70000, 4, dtype=torch.float32, device=flag_audio.device)
    y = torch.randn(1, 4, dtype=torch.float32, device=flag_audio.device)
    ref_out = torchaudio.functional.convolve(x, y, "full")
    res_out = flag_audio.ops.convolve(x, y, "full")
    gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)


@pytest.mark.convolve
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_accuracy_convolve_mode_as_kwarg(mode):
    # benchmark 通过 kwargs 传 mode（字符串无法经 unpack_to_args_kwargs 位置传递）
    x = torch.randn(2, 40, dtype=torch.float32, device=flag_audio.device)
    y = torch.randn(1, 5, dtype=torch.float32, device=flag_audio.device)
    res_out = flag_audio.ops.convolve(x, y, mode=mode)
    ref_out = torchaudio.functional.convolve(x, y, mode)
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
            x.double().cpu().numpy()[0],
            y.double().cpu().numpy()[0],
            mode="full",
        )
    )
    got = flag_audio.ops.convolve(x, y, "full").cpu().double()[0]
    rel_err = (got - golden).abs().max() / golden.abs().max()
    assert rel_err < 1e-3, f"大幅度下相对误差过大: {rel_err:.3e}"


def _expected_kaiming_std(shape, mode, nonlinearity, a):
    receptive_field = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    fan_in = shape[1] * receptive_field
    fan_out = shape[0] * receptive_field
    fan = fan_in if mode == "fan_in" else fan_out
    if nonlinearity == "leaky_relu":
        slope = 0.01 if a is None else a
        gain = (2.0 / (1.0 + slope**2)) ** 0.5
    elif nonlinearity == "relu":
        gain = 2.0**0.5
    elif nonlinearity == "tanh":
        gain = 5.0 / 3.0
    elif nonlinearity == "selu":
        gain = 3.0 / 4.0
    else:
        gain = 1.0
    return gain / fan**0.5


def _assert_kaiming_distribution(t, mode, nonlinearity, a):
    values = t.float()
    expected_std = _expected_kaiming_std(t.shape, mode, nonlinearity, a)
    assert torch.isfinite(values).all()
    assert abs(values.mean().item()) < expected_std * 0.05
    assert abs(values.std().item() - expected_std) / expected_std < 0.03


@pytest.mark.kaiming_normal_
@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (1, 1),
        (512, 256),
        (64, 32, 3, 3),
        (16, 8, 2, 2, 2),
        (503808, 1),
        (503809, 1),
        (1007617, 1),
        (4096, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["fan_in", "fan_out"])
@pytest.mark.parametrize("nonlinearity", ["leaky_relu", "relu", "tanh"])
def test_accuracy_kaiming_normal_(shape, dtype, mode, nonlinearity):
    res_t = torch.empty(shape, dtype=dtype, device=flag_audio.device)
    replay_t = torch.empty(shape, dtype=dtype, device=flag_audio.device)
    generator = torch.Generator(device=flag_audio.device).manual_seed(2024)
    replay_generator = torch.Generator(device=flag_audio.device).manual_seed(2024)
    out = flag_audio.ops.kaiming_normal_(
        res_t, a=0.1, mode=mode, nonlinearity=nonlinearity, generator=generator
    )
    flag_audio.ops.kaiming_normal_(
        replay_t, a=0.1, mode=mode, nonlinearity=nonlinearity, generator=replay_generator
    )
    assert out is res_t
    assert torch.equal(res_t, replay_t)
    if res_t.numel() >= 100000:
        _assert_kaiming_distribution(res_t, mode, nonlinearity, 0.1)


@pytest.mark.kaiming_normal_
@pytest.mark.parametrize(
    "nonlinearity, a",
    [
        ("sigmoid", 0),
        ("linear", 0),
        ("conv2d", 0),
        ("conv_transpose2d", 0),
        ("selu", 0),
        ("leaky_relu", 0),
        ("leaky_relu", None),
        ("leaky_relu", 0.3),
        ("leaky_relu", 1),
    ],
)
def test_accuracy_kaiming_normal_gain(nonlinearity, a):
    t = torch.empty(512, 512, dtype=torch.float32, device=flag_audio.device)
    flag_audio.ops.kaiming_normal_(t, a=a, mode="fan_in", nonlinearity=nonlinearity)
    _assert_kaiming_distribution(t, "fan_in", nonlinearity, a)


@pytest.mark.kaiming_normal_
def test_accuracy_kaiming_normal_sequential_draws():
    actual = torch.empty(2, 1000003, dtype=torch.float32, device=flag_audio.device)
    replay = torch.empty(2, 1000003, dtype=torch.float32, device=flag_audio.device)
    generator = torch.Generator(device=flag_audio.device).manual_seed(7)
    replay_generator = torch.Generator(device=flag_audio.device).manual_seed(7)
    for i in range(3):
        flag_audio.ops.kaiming_normal_(
            actual[i : i + 1] if i < 2 else actual, generator=generator
        )
        flag_audio.ops.kaiming_normal_(
            replay[i : i + 1] if i < 2 else replay, generator=replay_generator
        )
    assert torch.equal(actual, replay)
    assert not torch.equal(actual[0], actual[1])


@pytest.mark.kaiming_normal_
def test_accuracy_kaiming_normal_generator():
    actual = torch.empty(64, 64, dtype=torch.float32, device=flag_audio.device)
    replay = torch.empty(64, 64, dtype=torch.float32, device=flag_audio.device)
    generator = torch.Generator(device=flag_audio.device).manual_seed(42)
    replay_generator = torch.Generator(device=flag_audio.device).manual_seed(42)
    flag_audio.ops.kaiming_normal_(actual, generator=generator)
    flag_audio.ops.kaiming_normal_(replay, generator=replay_generator)
    assert torch.equal(actual, replay)


@pytest.mark.kaiming_normal_
def test_accuracy_kaiming_normal_matches_flaggems_rng():
    from flag_gems.ops.normal import normal_distribution

    shape = (1000, 257)  # 非 BLOCK*4 整数倍，覆盖尾块
    reference = torch.empty(shape, dtype=torch.float32, device=flag_audio.device)
    actual = torch.empty(shape, dtype=torch.float32, device=flag_audio.device)
    reference_generator = torch.Generator(device=flag_audio.device).manual_seed(91)
    actual_generator = torch.Generator(device=flag_audio.device).manual_seed(91)
    normal_distribution(
        shape,
        flag_audio.device,
        generator=reference_generator,
        out=reference,
    )
    std = _expected_kaiming_std(shape, "fan_in", "leaky_relu", 0.1)
    reference.mul_(std)
    flag_audio.ops.kaiming_normal_(
        actual,
        a=0.1,
        mode="fan_in",
        nonlinearity="leaky_relu",
        generator=actual_generator,
    )
    assert torch.equal(actual, reference)
    assert torch.equal(actual_generator.get_state(), reference_generator.get_state())


@pytest.mark.kaiming_normal_
@pytest.mark.parametrize("seed", [0, 1, 2, 2**63 - 1])
def test_accuracy_kaiming_normal_seed_edge(seed):
    actual = torch.empty(64, 32, dtype=torch.float32, device=flag_audio.device)
    replay = torch.empty(64, 32, dtype=torch.float32, device=flag_audio.device)
    torch.manual_seed(seed)
    flag_audio.ops.kaiming_normal_(actual)
    torch.manual_seed(seed)
    flag_audio.ops.kaiming_normal_(replay)
    assert torch.equal(actual, replay)


@pytest.mark.kaiming_normal_
def test_accuracy_kaiming_normal_philox_counter_carry():
    low_generator = torch.Generator(device=flag_audio.device).manual_seed(123)
    high_generator = torch.Generator(device=flag_audio.device).manual_seed(123)
    if not hasattr(high_generator, "set_offset"):
        pytest.skip("generator does not expose set_offset")
    low_generator.set_offset(0)
    high_generator.set_offset(2**32 - 4)
    low = torch.empty(1, 1024, dtype=torch.float32, device=flag_audio.device)
    high = torch.empty(1, 1024, dtype=torch.float32, device=flag_audio.device)
    flag_audio.ops.kaiming_normal_(low, generator=low_generator)
    flag_audio.ops.kaiming_normal_(high, generator=high_generator)
    # 高 offset 的前 4 个 counter 耗尽低 32 位空间；后续 counter 必须进入 c1=1，
    # 不能重新使用 offset=0 时的 c1=0 随机序列。
    assert not torch.equal(high.reshape(-1)[4:132], low.reshape(-1)[:128])


@pytest.mark.kaiming_normal_
def test_accuracy_kaiming_normal_interleaved_rng():
    actual = torch.empty(64, 32, dtype=torch.float32, device=flag_audio.device)
    replay = torch.empty(64, 32, dtype=torch.float32, device=flag_audio.device)
    torch.manual_seed(77)
    actual_pre = torch.rand(1000, device=flag_audio.device)
    flag_audio.ops.kaiming_normal_(actual)
    actual_post = torch.rand(1000, device=flag_audio.device)
    torch.manual_seed(77)
    replay_pre = torch.rand(1000, device=flag_audio.device)
    flag_audio.ops.kaiming_normal_(replay)
    replay_post = torch.rand(1000, device=flag_audio.device)
    assert torch.equal(actual_pre, replay_pre)
    assert torch.equal(actual, replay)
    assert torch.equal(actual_post, replay_post)


@pytest.mark.kaiming_normal_
def test_accuracy_kaiming_normal_kwargs():
    positional = torch.empty(32, 16, dtype=torch.float32, device=flag_audio.device)
    kwargs = torch.empty(32, 16, dtype=torch.float32, device=flag_audio.device)
    g1 = torch.Generator(device=flag_audio.device).manual_seed(5)
    g2 = torch.Generator(device=flag_audio.device).manual_seed(5)
    flag_audio.ops.kaiming_normal_(positional, 0.2, "fan_out", "relu", generator=g1)
    flag_audio.ops.kaiming_normal_(
        kwargs, a=0.2, mode="fan_out", nonlinearity="relu", generator=g2
    )
    assert torch.equal(positional, kwargs)


@pytest.mark.kaiming_normal_
def test_accuracy_kaiming_normal_mode_case_insensitive():
    lowercase = torch.empty(32, 16, dtype=torch.float32, device=flag_audio.device)
    uppercase = torch.empty(32, 16, dtype=torch.float32, device=flag_audio.device)
    g1 = torch.Generator(device=flag_audio.device).manual_seed(11)
    g2 = torch.Generator(device=flag_audio.device).manual_seed(11)
    flag_audio.ops.kaiming_normal_(lowercase, mode="fan_out", generator=g1)
    flag_audio.ops.kaiming_normal_(uppercase, mode="FAN_OUT", generator=g2)
    assert torch.equal(lowercase, uppercase)


@pytest.mark.kaiming_normal_
def test_kaiming_normal_invalid_args():
    t = torch.empty(2, 3, device=flag_audio.device)
    # 非法 mode
    with pytest.raises(ValueError):
        flag_audio.ops.kaiming_normal_(t, mode="bad_mode")
    # 非法 nonlinearity
    with pytest.raises(ValueError):
        flag_audio.ops.kaiming_normal_(t, nonlinearity="bad_act")
    # leaky_relu 的 a 不是合法数值（torch 对 bool 也报错）
    with pytest.raises(ValueError):
        flag_audio.ops.kaiming_normal_(t, a=True)
    with pytest.raises(ValueError):
        flag_audio.ops.kaiming_normal_(t, a="0.1")
    # ndim < 2 无法计算 fan（与 torch 行为一致）
    with pytest.raises(ValueError):
        flag_audio.ops.kaiming_normal_(torch.empty(8, device=flag_audio.device))


@pytest.mark.kaiming_normal_
def test_kaiming_normal_unsupported_inputs():
    t = torch.empty(2, 3, device=flag_audio.device)
    # fp64：torch 走 curand_normal2_double 另一路径，暂未复刻
    with pytest.raises(ValueError):
        flag_audio.ops.kaiming_normal_(t.to(torch.float64))
    # 非浮点 dtype
    with pytest.raises(ValueError):
        flag_audio.ops.kaiming_normal_(t.to(torch.int32))
    # complex dtype
    with pytest.raises(ValueError):
        flag_audio.ops.kaiming_normal_(t.to(torch.complex64))
    # requires_grad：暂不支持 autograd，必须显式报错而非静默丢梯度
    with pytest.raises(ValueError):
        flag_audio.ops.kaiming_normal_(t.clone().requires_grad_())
    # 非连续：in-place 算子无法靠 contiguous() 拷贝兜底，显式拒绝
    with pytest.raises(ValueError):
        flag_audio.ops.kaiming_normal_(torch.empty(3, 2, device=flag_audio.device).t())
    # int32 元素偏移上限：用 meta tensor 验证，避免实际分配 8GiB 显存
    huge = torch.empty(2**31, dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="2\\*\\*31-1"):
        flag_audio.ops.kaiming_normal_(huge)


@pytest.mark.kaiming_normal_
def test_kaiming_normal_empty_tensor():
    # 0 元素 tensor：与 torch 一样告警并直接返回
    t = torch.empty(0, 3, device=flag_audio.device)
    with pytest.warns(UserWarning):
        out = flag_audio.ops.kaiming_normal_(t)
    assert out.shape == (0, 3)


@pytest.mark.kaiming_normal_
@pytest.mark.parametrize("mode, nonlinearity", [("fan_in", "leaky_relu"), ("fan_out", "relu")])
def test_kaiming_normal_statistics(mode, nonlinearity):
    # 统计健全性：大样本均值≈0、std≈gain/sqrt(fan)
    t = torch.empty(1024, 1024, dtype=torch.float32, device=flag_audio.device)
    flag_audio.ops.kaiming_normal_(t, a=0, mode=mode, nonlinearity=nonlinearity)
    fan_in, fan_out = 1024, 1024
    fan = fan_in if mode == "fan_in" else fan_out
    gain = 2.0**0.5  # leaky_relu(a=0) 与 relu 的 gain 都是 sqrt(2)
    expected_std = gain / fan**0.5
    assert abs(t.mean().item()) < expected_std * 0.05
    assert abs(t.std().item() - expected_std) / expected_std < 0.01
