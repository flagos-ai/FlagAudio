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
import torchaudio

import flag_audio
from flag_audio.ops.psd import psd as gems_psd

from . import accuracy_utils as utils


def _ref_psd(specgram, mask=None, normalize=True, eps=1e-10):
    return torchaudio.functional.psd(
        specgram, mask=mask, normalize=normalize, eps=eps
    )


@pytest.mark.psd
@pytest.mark.parametrize("shape", [(2, 3, 5, 7), (1, 1, 9, 11), (4, 6, 257, 100)])
@pytest.mark.parametrize("has_mask", [False, True])
@pytest.mark.parametrize("normalize", [True, False])
def test_psd_basic(shape, has_mask, normalize):
    b, ch, freq, time = shape
    specgram = torch.randn(
        b, ch, freq, time, device=flag_audio.device, dtype=torch.complex64
    )
    mask = (
        torch.rand(b, freq, time, device=flag_audio.device) if has_mask else None
    )

    ref = _ref_psd(specgram, mask=mask, normalize=normalize)
    res = gems_psd(specgram, mask=mask, normalize=normalize)

    assert res.shape == ref.shape, f"Shape mismatch: {res.shape} vs {ref.shape}"
    utils.gems_assert_close(res, ref, torch.complex64, atol=1e-4)


@pytest.mark.psd
@pytest.mark.parametrize("shape", [(2, 3, 4, 5, 7)])
def test_psd_multidim_batch(shape):
    specgram = torch.randn(
        *shape, device=flag_audio.device, dtype=torch.complex64
    )
    ref = _ref_psd(specgram)
    res = gems_psd(specgram)
    assert res.shape == ref.shape
    utils.gems_assert_close(res, ref, torch.complex64, atol=1e-4)


@pytest.mark.psd
def test_psd_hermitian():
    specgram = torch.randn(
        2, 4, 33, 50, device=flag_audio.device, dtype=torch.complex64
    )
    res = gems_psd(specgram)
    # PSD must be Hermitian: PSD[f] = PSD[f]^H
    diff = (res - res.conj().transpose(-1, -2)).abs().max().item()
    assert diff < 1e-4, f"PSD not Hermitian, max diff={diff}"


@pytest.mark.psd
def test_psd_non_contiguous():
    specgram = torch.randn(
        2, 3, 5, 7, device=flag_audio.device, dtype=torch.complex64
    )
    # Make the channel/freq/time part non-contiguous.
    spec_nc = specgram.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
    assert not spec_nc.is_contiguous()

    ref = _ref_psd(spec_nc)
    res = gems_psd(spec_nc)
    utils.gems_assert_close(res, ref, torch.complex64, atol=1e-4)


@pytest.mark.psd
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_psd_dtype(dtype):
    specgram = torch.randn(2, 3, 5, 7, device=flag_audio.device, dtype=dtype)
    mask = torch.rand(2, 5, 7, device=flag_audio.device)
    ref = _ref_psd(specgram, mask=mask, normalize=True)
    res = gems_psd(specgram, mask=mask, normalize=True)
    assert res.dtype == dtype
    # complex128 has a slightly looser tolerance because the Triton
    # reduction order differs from torch.einsum (relative error ~1e-8).
    atol = 1e-4 if dtype == torch.complex64 else 1e-6
    assert torch.allclose(res, ref, atol=atol, rtol=atol)


@pytest.mark.psd
@pytest.mark.parametrize(
    "spec_dtype,mask_dtype",
    [
        (torch.complex64, torch.float32),
        (torch.complex64, torch.float64),
        (torch.complex128, torch.float32),
        (torch.complex128, torch.float64),
        (torch.complex64, torch.int32),
        (torch.complex64, torch.int64),
    ],
)
def test_psd_mask_dtype_promotion(spec_dtype, mask_dtype):
    specgram = torch.randn(2, 3, 5, 7, device=flag_audio.device, dtype=spec_dtype)
    mask = torch.rand(2, 5, 7, device=flag_audio.device).to(mask_dtype)
    ref = _ref_psd(specgram, mask=mask, normalize=True)
    res = gems_psd(specgram, mask=mask, normalize=True)
    # Output dtype must follow PyTorch type promotion (psd * mask), matching
    # torchaudio exactly.
    assert res.dtype == ref.dtype
    atol = 1e-4 if res.dtype == torch.complex64 else 1e-6
    assert torch.allclose(res, ref, atol=atol, rtol=atol)


@pytest.mark.psd
@pytest.mark.parametrize("eps", [1e-10, 1e-8, 1e-5])
def test_psd_eps(eps):
    specgram = torch.randn(2, 3, 5, 7, device=flag_audio.device, dtype=torch.complex64)
    mask = torch.rand(2, 5, 7, device=flag_audio.device)
    ref = _ref_psd(specgram, mask=mask, normalize=True, eps=eps)
    res = gems_psd(specgram, mask=mask, normalize=True, eps=eps)
    utils.gems_assert_close(res, ref, torch.complex64, atol=1e-4)


@pytest.mark.psd
def test_psd_default_eps():
    specgram = torch.randn(2, 3, 5, 7, device=flag_audio.device, dtype=torch.complex64)
    mask = torch.full((2, 5, 7), 1e-8, device=flag_audio.device)
    ref = _ref_psd(specgram, mask=mask)
    res = gems_psd(specgram, mask=mask)
    utils.gems_assert_close(res, ref, torch.complex64, atol=1e-4)


@pytest.mark.psd
@pytest.mark.parametrize("shape", [(0, 3, 5, 7), (2, 0, 5, 7), (2, 3, 0, 7), (2, 3, 5, 0)])
def test_psd_empty(shape):
    specgram = torch.randn(*shape, device=flag_audio.device, dtype=torch.complex64)
    ref = _ref_psd(specgram)
    res = gems_psd(specgram)
    assert res.shape == ref.shape
    # torchaudio sums over the empty axis -> all-zero PSD; never uninitialized.
    assert torch.all(res == 0).item()
    assert torch.all(ref == 0).item()


@pytest.mark.psd
def test_psd_errors():
    # Not a tensor
    with pytest.raises(TypeError):
        gems_psd([1, 2, 3])

    # Too few dimensions
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        gems_psd(torch.randn(2, 3, device=flag_audio.device, dtype=torch.complex64))

    # Not complex
    with pytest.raises(ValueError, match="complex-valued"):
        gems_psd(torch.randn(2, 3, 5, 7, device=flag_audio.device))

    # Invalid mask type
    specgram = torch.randn(2, 3, 5, 7, device=flag_audio.device, dtype=torch.complex64)
    with pytest.raises(TypeError, match="mask must be a torch.Tensor"):
        gems_psd(specgram, mask=[1, 2, 3])

    # Wrong mask shape
    with pytest.raises(ValueError, match="dimensions of mask"):
        gems_psd(specgram, mask=torch.rand(2, 4, 7, device=flag_audio.device))

    # Autograd is not supported: must raise instead of silently dropping grad.
    specgram_grad = torch.randn(
        2, 3, 5, 7, device=flag_audio.device, dtype=torch.complex64, requires_grad=True
    )
    with pytest.raises(ValueError, match="autograd"):
        gems_psd(specgram_grad)
    mask_grad = torch.rand(2, 5, 7, device=flag_audio.device, requires_grad=True)
    with pytest.raises(ValueError, match="autograd"):
        gems_psd(specgram.detach(), mask=mask_grad)
