import math
import os
from typing import Any, Generator, List, Optional, Tuple

import pytest
import torch
import triton
import torchaudio
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import gain_triton,DB_to_amplitude_triton, mask_along_axis_iid_triton,mask_along_axis_triton, preemphasis_triton

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
        op_name="scaled_dot_product_attention",
        input_fn=gain_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(gain_triton)
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
    bench.set_gems(DB_to_amplitude_triton)
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
    bench.set_gems(mask_along_axis_iid_triton)
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
    bench.set_gems(mask_along_axis_triton)
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
    bench.set_gems(preemphasis_triton)
    bench.run()


# from benchmark.attri_util import FLOAT_DTYPES
# from benchmark.performance_utils import (
#     GenericBenchmark,
#     GenericBenchmarkExcluse1D,
#     binary_input_fn,
# )

# from .performance_utils import Benchmark, SkipVersion


# @pytest.mark.skip_layernorm
# def test_perf_skip_layernorm():
#     def skip_layernorm_input_fn(shape, dtype, device):
#         inp = torch.randn(shape, dtype=dtype, device=device)
#         residual = torch.randn(shape, dtype=dtype, device=device)
#         layer_shape = (shape[-1],)
#         weight = torch.randn(layer_shape, dtype=dtype, device=device)
#         bias = torch.randn(layer_shape, dtype=dtype, device=device)
#         yield inp, residual, layer_shape, weight, bias

#     def torch_op(inp, residual, layer_shape, weight, bias):
#         return torch.layer_norm(inp + residual, layer_shape, weight, bias)

#     gems_op = skip_layer_norm

#     bench = GenericBenchmarkExcluse1D(
#         input_fn=skip_layernorm_input_fn,
#         op_name="skip_layernorm",
#         torch_op=torch_op,
#         dtypes=FLOAT_DTYPES,
#     )
#     bench.set_gems(gems_op)
#     bench.run()

# @pytest.mark.scaled_dot_product_attention
# @pytest.mark.parametrize("gain_db", [6.0])
# def test_perf_gain(gain_db):
#     def gain_input(shape, dtype, device):
#         print("device======", device)
#         print("dtyp=====",dtype)
#         print("shape====",shape)
#         input = torch.randn(shape, device=device, dtype=dtype)
#         yield input, gain_db

#     bench = TorchaudioBenchmark(
#         op_name="scaled_dot_product_attention",
#         input_fn=gain_input,
#         # torch_op=torch.nn.functional.scaled_dot_product_attention,
#         torch_op=torchaudio.functional.gain,
#         dtypes=[
#             torch.float32,
#         ],
#     )
#     bench.set_gems(DB_to_amplitude_triton)
#     bench.run()