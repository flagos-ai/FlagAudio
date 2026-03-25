import math
import os
from typing import Any, Generator, List, Optional, Tuple

import pytest
import torch
import triton
import torchaudio
import sys

import flag_audio

from .performance_utils import Benchmark, GenericBenchmark

class TorchaudioBenchmark(GenericBenchmark):
    """
    benchmark for attention
    """

    def set_more_shapes(self):
        return None


@pytest.mark.gain
@pytest.mark.parametrize("gain_db", [6.0])
def test_perf_gain(gain_db):
    def gain_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, gain_db
    def ref_op(input, gain_db):
        return torchaudio.functional.gain(input, gain_db)

    bench = TorchaudioBenchmark(
        op_name="gain",
        input_fn=gain_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.gain)
    bench.run()

@pytest.mark.DB_to_amplitude
@pytest.mark.parametrize("ref", [3.0])
@pytest.mark.parametrize("power", [0.5])
def test_perf_DB_to_amplitude(ref, power):
    def DB_to_amplitude_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, ref, power
    def ref_op(input, ref, power):
        return torchaudio.functional.DB_to_amplitude(input, ref, power)

    bench = TorchaudioBenchmark(
        op_name="DB_to_amplitude",
        input_fn=DB_to_amplitude_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.DB_to_amplitude)
    bench.run()


class MaskAlongBenchmark(GenericBenchmark):
    """
    benchmark for attention
    """

    def set_more_shapes(self):
        return None

@pytest.mark.mask_along_axis_iid
@pytest.mark.parametrize("mask_param", [3])
@pytest.mark.parametrize("mask_value", [0.0])
@pytest.mark.parametrize("axis", [3])
@pytest.mark.parametrize("p", [1])
def test_perf_mask_along_axis_iid(mask_param, mask_value, axis, p):
    def mask_along_axis_iid_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, mask_param, mask_value, axis, p
    def ref_op(input, mask_param, mask_value, axis, p):
        return torchaudio.functional.mask_along_axis_iid(input, mask_param, mask_value, axis, p)

    bench = MaskAlongBenchmark(
        op_name="mask_along_axis_iid",
        input_fn=mask_along_axis_iid_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.mask_along_axis_iid)
    bench.run()


@pytest.mark.mask_along_axis
@pytest.mark.parametrize("mask_param", [6])
@pytest.mark.parametrize("mask_value", [0.0])
@pytest.mark.parametrize("axis", [2])
@pytest.mark.parametrize("p", [0.5])
def test_perf_mask_along_axis(mask_param, mask_value, axis, p):
    def mask_along_axis_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, mask_param, mask_value, axis, p
    def ref_op(input, mask_param, mask_value, axis, p):
        return torchaudio.functional.mask_along_axis(input, mask_param, mask_value, axis, p)

    bench = MaskAlongBenchmark(
        op_name="mask_along_axis",
        input_fn=mask_along_axis_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.mask_along_axis)
    bench.run()

@pytest.mark.preemphasis
@pytest.mark.parametrize("coeff", [0.97])
def test_perf_preemphasis(coeff):
    def preemphasis_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, coeff
    def ref_op(input, coeff):
        return torchaudio.functional.preemphasis(input, coeff)

    bench = MaskAlongBenchmark(
        op_name="preemphasis",
        input_fn=preemphasis_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.preemphasis)
    bench.run()
