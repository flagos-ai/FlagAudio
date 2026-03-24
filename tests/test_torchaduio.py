import logging
import math
import os
import random
import sys

import numpy as np
import pytest
import torch

import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import gain_triton,DB_to_amplitude_triton,mask_along_axis_iid_triton,mask_along_axis_triton,preemphasis_triton

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    SCALARS,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import TO_CPU

device = "cuda"

@pytest.mark.gain
@pytest.mark.parametrize("shape", [(3)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("gain_db", [(6.0)])
def test_accuracy_gain(shape, gain_db, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.gain(ref_inp1_tensor, gain_db)

    res_out = gain_triton(input_tensor, gain_db)

    gems_assert_close(res_out, ref_out, dtype)



@pytest.mark.DB_to_amplitude
@pytest.mark.parametrize("shape", [(3)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("ref", [3.0])
@pytest.mark.parametrize("power", [0.5])
def test_accuracy_DB_to_amplitude(shape, dtype, ref, power):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.DB_to_amplitude(ref_inp1_tensor, ref, power)
    res_out = DB_to_amplitude_triton(input_tensor, ref, power)

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
def test_accuracy_mask_along_axis_iid(shape, dtype,mask_param, mask_value, axis, p):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    ref_inp1_tensor = to_reference(input_tensor, True)
    
    torch.rand = mock_rand
    ref_out = torchaudio.functional.mask_along_axis_iid(ref_inp1_tensor, mask_param, mask_value, axis, p)
    res_out = mask_along_axis_iid_triton(input_tensor, mask_param, mask_value, axis, p)

    gems_assert_close(res_out, ref_out, dtype)  

@pytest.mark.mask_along_axis
@pytest.mark.parametrize("shape", [(2,2,4,4)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("mask_param", [6])
@pytest.mark.parametrize("mask_value", [0.0])
@pytest.mark.parametrize("axis", [2])
@pytest.mark.parametrize("p", [0.5])
def test_accuracy_mask_along_axis(shape, dtype,mask_param, mask_value, axis, p):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    torch.rand = mock_rand
    ref_out = torchaudio.functional.mask_along_axis(ref_inp1_tensor, mask_param, mask_value, axis, p)
    res_out = mask_along_axis_triton(input_tensor, mask_param, mask_value, axis, p)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.preemphasis
@pytest.mark.parametrize("shape", [(2, 2)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("coeff", [0.97])
def test_accuracy_preemphasis(shape, coeff, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.preemphasis(ref_inp1_tensor, coeff)

    res_out = preemphasis_triton(input_tensor, coeff)

    gems_assert_close(res_out, ref_out, dtype)