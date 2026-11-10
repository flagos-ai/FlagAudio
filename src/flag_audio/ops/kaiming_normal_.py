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

import functools
import math
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.ops.randn import pair_uniform_to_normal_fast
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import uint_to_uniform_float

from flag_audio.utils.random_utils import philox_backend_seed_offset


@functools.lru_cache(maxsize=1024)
def _fan_from_shape(shape):
    # 与 torch.nn.init._calculate_fan_in_and_fan_out 一致
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer "
            f"than 2 dimensions (got {dimensions})."
        )
    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        for size in shape[2:]:
            receptive_field_size *= size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor):
    return _fan_from_shape(tuple(tensor.shape))


@functools.lru_cache(maxsize=128)
def _calculate_gain_cached(nonlinearity: str, a) -> float:
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        negative_slope = 0.01 if a is None else a
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _calculate_gain(nonlinearity: str, a) -> float:
    # 与 torch.nn.init.calculate_gain 一致；先校验，避免缓存未 hashable 的非法参数
    if nonlinearity == "leaky_relu":
        if a is not None and (
            isinstance(a, bool) or not isinstance(a, (int, float))
        ):
            raise ValueError(f"negative_slope {a} not a valid number")
        return _calculate_gain_cached(nonlinearity, a)
    return _calculate_gain_cached(nonlinearity, None)


def _check_tensor_supported(tensor: torch.Tensor) -> None:
    if tensor.is_complex():
        raise ValueError(f"kaiming_normal_ does not support complex dtypes (got {tensor.dtype}).")
    if not tensor.dtype.is_floating_point:
        raise ValueError(
            f"kaiming_normal_ expects a floating point tensor (got {tensor.dtype})."
        )
    if tensor.dtype == torch.float64:
        raise ValueError(
            "kaiming_normal_ does not support float64 yet; use float32/float16/bfloat16."
        )
    if tensor.requires_grad:
        raise ValueError(
            "kaiming_normal_ does not support autograd yet; detach the tensor before "
            "calling to avoid silently losing gradients."
        )
    if not tensor.is_contiguous():
        # in-place 算子不能通过 contiguous() 拷贝兜底（结果写不回去），显式拒绝
        raise ValueError("kaiming_normal_ expects a contiguous tensor.")
    if tensor.numel() > 2**31 - 1:
        raise ValueError(
            f"kaiming_normal_ supports at most 2**31-1 elements (got {tensor.numel()})."
        )


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def kaiming_normal_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    std,
    BLOCK: tl.constexpr = 512,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.uint32)
    c0 = c0_base + i4
    # uint32 回绕时必须向高 32 位进位，否则高 offset 下会重复使用低 counter 序列
    c1 += (c0 < c0_base).to(tl.uint32)
    zero = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, zero, zero)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    n0, n1 = pair_uniform_to_normal_fast(r0, r1)
    n2, n3 = pair_uniform_to_normal_fast(r2, r3)

    off0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK
    tl.store(out_ptr + off0, n0 * std, mask=off0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off1, n1 * std, mask=off1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off2, n2 * std, mask=off2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off3, n3 * std, mask=off3 < N, eviction_policy="evict_first")


def _launch_normal_kernel(tensor, std, generator, numel, block, grid):
    increment = triton.cdiv(numel, 4)
    seed, offset = philox_backend_seed_offset(increment, generator=generator)
    kaiming_normal_kernel[grid](
        tensor,
        numel,
        seed,
        offset,
        std,
        BLOCK=block,
    )


def _normal_fill_(tensor: torch.Tensor, std: float, generator) -> None:
    """单次通用 Triton kernel 完成正态生成、std 缩放和写出。"""
    numel = tensor.numel()
    block = 512
    grid = (triton.cdiv(numel, block * 4),)
    if tensor.device.index == torch_device_fn.current_device():
        _launch_normal_kernel(tensor, std, generator, numel, block, grid)
    else:
        with torch_device_fn.device(tensor.device):
            _launch_normal_kernel(tensor, std, generator, numel, block, grid)


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    *,
    generator=None,
) -> torch.Tensor:
    """用 Kaiming 正态分布原位初始化 tensor。

    通过单个 FlagGems 通用 Triton kernel 完成 Philox、正态变换、标准差缩放
    和写出。保证初始化分布、参数语义和生成器状态推进，不要求与任何厂商的
    原生 torch 随机序列逐位一致。

    暂不支持 float64、autograd（requires_grad 会报错而非静默丢失梯度）和
    非连续 tensor。
    """
    _check_tensor_supported(tensor)
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor

    # 与 torch._calculate_correct_fan 一致：mode 先小写化再校验
    mode = mode.lower()
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(
            f"Mode {mode} not supported, please use one of ['fan_in', 'fan_out']"
        )
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out

    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    _normal_fill_(tensor, std, generator)
    return tensor


def test_op():
    import flag_audio

    device = flag_audio.device
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        t = torch.empty(1024, 1024, dtype=dtype, device=device)
        result = kaiming_normal_(t, a=0, mode="fan_in", nonlinearity="leaky_relu")
        assert result is t
        assert torch.isfinite(t).all()
        values = t.float()
        expected_std = math.sqrt(2.0) / math.sqrt(1024)
        assert abs(values.mean().item()) < expected_std * 0.05
        assert abs(values.std().item() - expected_std) / expected_std < 0.01

    # 同 seed 可复现，连续抽取不重复
    g1 = torch.Generator(device=device).manual_seed(1234)
    g2 = torch.Generator(device=device).manual_seed(1234)
    t1 = torch.empty(64, 32, device=device)
    t2 = torch.empty(64, 32, device=device)
    kaiming_normal_(t1, generator=g1)
    kaiming_normal_(t2, generator=g2)
    assert torch.equal(t1, t2)
    kaiming_normal_(t2, generator=g1)
    assert not torch.equal(t1, t2)

    # 非法参数
    for bad_call in (
        lambda: kaiming_normal_(torch.empty(8, device=device)),
        lambda: kaiming_normal_(torch.empty(2, 3, device=device), mode="bad"),
        lambda: kaiming_normal_(torch.empty(2, 3, device=device), nonlinearity="bad"),
        lambda: kaiming_normal_(torch.empty(2, 3, dtype=torch.float64, device=device)),
        lambda: kaiming_normal_(torch.empty(2, 3, device=device).t()),
    ):
        try:
            bad_call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


if __name__ == "__main__":
    test_op()
