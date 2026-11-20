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

import importlib

import pytest
import torch
import torchaudio

import flag_audio


pytestmark = pytest.mark.rtf_evd

_RTF_DTYPES = [
    torch.complex64,
    pytest.param(
        torch.complex128,
        marks=pytest.mark.skipif(
            not flag_audio.runtime.device.support_fp64,
            reason="The current backend does not support FP64.",
        ),
    ),
]


def _random_psd(shape, dtype, rank=None):
    channel = shape[-1]
    rank = channel if rank is None else rank
    generator = torch.randn(
        *shape[:-2], channel, rank, device=flag_audio.device, dtype=dtype
    )
    psd = generator @ generator.conj().transpose(-1, -2) / max(rank, 1)
    return psd


def _hermitian_with_spectrum(eigenvalues, dtype, seed=0):
    torch.manual_seed(seed)
    channel = eigenvalues.numel()
    basis_seed = torch.randn(channel, channel, device=flag_audio.device, dtype=dtype)
    basis = torch.linalg.qr(basis_seed).Q
    diagonal = torch.diag(eigenvalues).to(dtype)
    return basis @ diagonal @ basis.conj().T


def _assert_principal_eigenvector(psd, actual, check_direction=True):
    expected = torchaudio.functional.rtf_evd(psd)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype

    real_dtype = torch.float64 if psd.dtype == torch.complex128 else torch.float32
    tolerance = 5e-12 if psd.dtype == torch.complex128 else 8e-6
    unit_error = (actual.abs().square().sum(-1) - 1.0).abs()
    torch.testing.assert_close(
        unit_error,
        torch.zeros_like(unit_error),
        atol=tolerance,
        rtol=0.0,
    )

    lower = torch.tril(psd)
    effective_psd = lower + torch.triu(lower.conj().transpose(-1, -2), diagonal=1)
    effective_psd = effective_psd.clone()
    effective_psd.diagonal(dim1=-2, dim2=-1).imag.zero_()
    scale = effective_psd.abs().sum(-1).max(dim=-1).values
    normalized_psd = effective_psd / scale.clamp_min(
        torch.finfo(real_dtype).tiny
    ).unsqueeze(-1).unsqueeze(-1)
    product = (normalized_psd @ actual.unsqueeze(-1)).squeeze(-1)
    eigenvalue = (actual.conj() * product).sum(-1).real
    residual = product - eigenvalue.to(psd.dtype).unsqueeze(-1) * actual
    residual_norm = residual.abs().square().sum(-1).sqrt()
    torch.testing.assert_close(
        residual_norm,
        torch.zeros_like(residual_norm),
        atol=tolerance,
        rtol=0.0,
    )

    expected_eigenvalue = torch.linalg.eigvalsh(normalized_psd)[..., -1]
    relative_eigenvalue_error = (eigenvalue - expected_eigenvalue).abs() / (
        expected_eigenvalue.abs().clamp_min(torch.finfo(real_dtype).tiny)
    )
    torch.testing.assert_close(
        relative_eigenvalue_error,
        torch.zeros_like(relative_eigenvalue_error),
        atol=tolerance,
        rtol=0.0,
    )

    if check_direction:
        similarity = (expected.conj() * actual).sum(-1).abs()
        torch.testing.assert_close(
            similarity,
            torch.ones_like(similarity, dtype=real_dtype),
            atol=2e-4 if psd.dtype == torch.complex64 else 1e-8,
            rtol=0.0,
        )


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (4, 4),
        (2, 7, 4, 4),
        (2, 3, 5, 8, 8),
        (2, 17, 16, 16),
        (1, 3, 32, 32),
        (1, 2, 33, 33),
        (1, 2, 64, 64),
        (1, 1, 65, 65),
    ],
)
def test_random_psd(shape, dtype):
    psd = _random_psd(shape, dtype)
    actual = flag_audio.ops.rtf_evd(psd)
    _assert_principal_eigenvector(psd, actual)


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
@pytest.mark.parametrize("channel", [2, 4, 8])
def test_rank_one_psd(dtype, channel):
    psd = _random_psd((2, 33, channel, channel), dtype, rank=1)
    actual = flag_audio.ops.rtf_evd(psd)
    _assert_principal_eigenvector(psd, actual)


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
def test_zero_psd(dtype):
    psd = torch.zeros(2, 5, 4, 4, device=flag_audio.device, dtype=dtype)
    actual = flag_audio.ops.rtf_evd(psd)
    _assert_principal_eigenvector(psd, actual, check_direction=False)
    torch.testing.assert_close(actual[..., 0], torch.ones_like(actual[..., 0]))
    torch.testing.assert_close(actual[..., 1:], torch.zeros_like(actual[..., 1:]))


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
def test_repeated_principal_eigenvalue(dtype):
    eigenvalues = torch.diag(
        torch.tensor([1.0, 1.0, 0.5, 0.25], device=flag_audio.device)
    ).to(dtype)
    seed = torch.randn(4, 4, device=flag_audio.device, dtype=dtype)
    basis = torch.linalg.qr(seed).Q
    psd = basis @ eigenvalues @ basis.conj().T
    actual = flag_audio.ops.rtf_evd(psd)
    _assert_principal_eigenvector(psd, actual, check_direction=False)


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
def test_close_principal_eigenvalues(dtype):
    eigenvalues = torch.diag(
        torch.tensor([1.0, 0.993, 0.4, 0.2], device=flag_audio.device)
    ).to(dtype)
    seed = torch.randn(4, 4, device=flag_audio.device, dtype=dtype)
    basis = torch.linalg.qr(seed).Q
    psd = basis @ eigenvalues @ basis.conj().T
    actual = flag_audio.ops.rtf_evd(psd)
    _assert_principal_eigenvector(psd, actual)


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
@pytest.mark.parametrize("channel", [4, 33])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_extreme_spectral_gap_multi_seed(dtype, channel, seed):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    second = 0.999999999999 if dtype == torch.complex128 else 0.999999
    eigenvalues = torch.full((channel,), 0.1, device=flag_audio.device, dtype=real_dtype)
    eigenvalues[0] = 1.0
    eigenvalues[1] = second
    matrix = _hermitian_with_spectrum(eigenvalues, dtype, seed)
    actual = flag_audio.ops.rtf_evd(matrix)
    _assert_principal_eigenvector(matrix, actual, check_direction=False)


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
@pytest.mark.parametrize("channel", [4, 33])
@pytest.mark.parametrize("scale", [-30, 30])
def test_extreme_matrix_scale(dtype, channel, scale):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    exponent = scale if dtype == torch.complex64 else (200 if scale > 0 else -200)
    eigenvalues = torch.linspace(0.1, 1.0, channel, device=flag_audio.device, dtype=real_dtype)
    matrix = _hermitian_with_spectrum(eigenvalues, dtype) * (10.0**exponent)
    actual = flag_audio.ops.rtf_evd(matrix)
    _assert_principal_eigenvector(matrix, actual)


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
@pytest.mark.parametrize("channel", [4, 33])
def test_nonconvergence_returns_nan(dtype, channel, monkeypatch):
    module = importlib.import_module("flag_audio.ops.rtf_evd")
    monkeypatch.setattr(module, "_POWER_ITER", 0)
    monkeypatch.setattr(module, "_SQUARE_ITER", 0)
    matrix = _random_psd((channel, channel), dtype)
    actual = flag_audio.ops.rtf_evd(matrix)
    assert torch.isnan(actual).all()


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
def test_general_hermitian_with_gershgorin_shift(dtype):
    seed = torch.randn(2, 3, 4, 4, device=flag_audio.device, dtype=dtype)
    psd = (seed + seed.conj().transpose(-1, -2)) / 2
    actual = flag_audio.ops.rtf_evd(psd)
    _assert_principal_eigenvector(psd, actual)


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
@pytest.mark.parametrize("channel", [4, 33])
def test_noncontiguous_and_conjugate_view(dtype, channel):
    psd = _random_psd((2, 5, channel, channel), dtype)
    psd = psd.transpose(-1, -2).conj()
    assert not psd.is_contiguous()
    actual = flag_audio.ops.rtf_evd(psd)
    _assert_principal_eigenvector(psd, actual)


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
def test_noncontiguous_batch_strides(dtype):
    psd = _random_psd((2, 3, 5, 4, 4), dtype)
    psd = psd.transpose(0, 1)
    assert not psd.is_contiguous()
    actual = flag_audio.ops.rtf_evd(psd)
    _assert_principal_eigenvector(psd, actual)


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
def test_only_lower_triangle_is_read(dtype):
    psd = _random_psd((2, 3, 4, 4), dtype)
    corrupted = psd.clone()
    upper = torch.triu(torch.ones_like(corrupted), diagonal=1).to(torch.bool)
    corrupted[upper] = 100 + 20j
    actual = flag_audio.ops.rtf_evd(corrupted)
    _assert_principal_eigenvector(corrupted, actual)


@pytest.mark.parametrize(
    "shape",
    [(0, 4, 4), (2, 0, 4, 4)],
)
def test_empty_input(shape):
    psd = torch.empty(shape, device=flag_audio.device, dtype=torch.complex64)
    actual = flag_audio.ops.rtf_evd(psd)
    expected = torchaudio.functional.rtf_evd(psd)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.numel() == 0


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
def test_negative_dominant_hermitian(dtype):
    eigenvalues = torch.diag(
        torch.tensor([-5.0, 2.0, 1.0, 0.5], device=flag_audio.device)
    ).to(dtype)
    seed = torch.randn(4, 4, device=flag_audio.device, dtype=dtype)
    basis = torch.linalg.qr(seed).Q
    matrix = basis @ eigenvalues @ basis.conj().T
    actual = flag_audio.ops.rtf_evd(matrix)
    _assert_principal_eigenvector(matrix, actual)


@pytest.mark.parametrize("dtype", _RTF_DTYPES)
@pytest.mark.parametrize("channel", [4, 33])
def test_negative_definite_matrix(dtype, channel):
    diagonal = torch.linspace(-4.0, -1.0, channel, device=flag_audio.device)
    matrix = torch.diag(diagonal).to(dtype)
    actual = flag_audio.ops.rtf_evd(matrix)
    _assert_principal_eigenvector(matrix, actual)
    expected = torch.zeros(channel, device=flag_audio.device, dtype=dtype)
    expected[-1] = 1.0
    torch.testing.assert_close(actual, expected)


def test_known_diagonal_matrix():
    diagonal = torch.tensor([0.5, 3.0, 2.0], device=flag_audio.device)
    psd = torch.diag(diagonal).to(torch.complex64)
    actual = flag_audio.ops.rtf_evd(psd)
    expected = torch.zeros(3, device=flag_audio.device, dtype=torch.complex64)
    expected[1] = 1.0
    torch.testing.assert_close(actual, expected)


def test_invalid_inputs():
    with pytest.raises(TypeError, match="torch.cfloat"):
        flag_audio.ops.rtf_evd(torch.ones(4, 4, device=flag_audio.device))
    with pytest.raises(ValueError, match="last two dimensions"):
        flag_audio.ops.rtf_evd(
            torch.ones(2, 3, device=flag_audio.device, dtype=torch.complex64)
        )
    with pytest.raises(ValueError, match="at least two dimensions"):
        flag_audio.ops.rtf_evd(
            torch.ones(4, device=flag_audio.device, dtype=torch.complex64)
        )
    with pytest.raises(IndexError, match="out of bounds"):
        flag_audio.ops.rtf_evd(
            torch.empty(2, 3, 0, 0, device=flag_audio.device, dtype=torch.complex64)
        )
    with pytest.raises(RuntimeError, match="autograd"):
        flag_audio.ops.rtf_evd(
            torch.eye(
                4, device=flag_audio.device, dtype=torch.complex64, requires_grad=True
            )
        )
