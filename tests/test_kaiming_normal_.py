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
