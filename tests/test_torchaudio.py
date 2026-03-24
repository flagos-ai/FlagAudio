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
@pytest.mark.parametrize("gain_db", [(6.0)])
def test_accuracy_gain(shape, gain_db, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.gain(ref_inp1_tensor, gain_db)

    res_out = flag_audio.ops.gain_triton(input_tensor, gain_db)

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
    res_out = flag_audio.ops.DB_to_amplitude_triton(input_tensor, ref, power)

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
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)
    
    torch.rand = mock_rand
    ref_out = torchaudio.functional.mask_along_axis_iid(ref_inp1_tensor, mask_param, mask_value, axis, p)
    res_out = flag_audio.ops.mask_along_axis_iid_triton(input_tensor, mask_param, mask_value, axis, p)

    gems_assert_close(res_out, ref_out, dtype)  

@pytest.mark.mask_along_axis
@pytest.mark.parametrize("shape", [(2,2,4,4)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("mask_param", [6])
@pytest.mark.parametrize("mask_value", [0.0])
@pytest.mark.parametrize("axis", [2])
@pytest.mark.parametrize("p", [0.5])
def test_accuracy_mask_along_axis(shape, dtype,mask_param, mask_value, axis, p):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    torch.rand = mock_rand
    ref_out = torchaudio.functional.mask_along_axis(ref_inp1_tensor, mask_param, mask_value, axis, p)
    res_out = flag_audio.ops.mask_along_axis_triton(input_tensor, mask_param, mask_value, axis, p)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.preemphasis
@pytest.mark.parametrize("shape", [(2, 2)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("coeff", [0.97])
def test_accuracy_preemphasis(shape, coeff, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.preemphasis(ref_inp1_tensor, coeff)

    res_out = flag_audio.ops.preemphasis_triton(input_tensor, coeff)

    gems_assert_close(res_out, ref_out, dtype)


