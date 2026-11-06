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

"""Accuracy tests for the Triton ``inverse_spectrogram`` op.

Reference: ``torchaudio.functional.inverse_spectrogram``. These tests run on
whatever device ``flag_gems.device`` resolves to (cuda / musa / npu / cpu),
so the same suite exercises CUDA, MUSA and Ascend backends.
"""

import pytest
import torch
import torchaudio

import flag_gems
from flag_audio.ops.inverse_spectrogram import inverse_spectrogram as gems_istft

from . import accuracy_utils as utils


def _forward_spec(waveform, n_fft, hop_length, win_length, power, normalized, center, onesided):
    window = torch.hann_window(win_length, device=waveform.device)
    return torchaudio.functional.spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        center=center,
        pad_mode="reflect",
        onesided=onesided,
    )


def _ref_istft(spec, length, window, n_fft, hop_length, win_length, normalized, center, onesided):
    return torchaudio.functional.inverse_spectrogram(
        spectrogram=spec,
        length=length,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        normalized=normalized,
        center=center,
        pad_mode="reflect",
        onesided=onesided,
    )


def _gems_istft(spec, length, window, n_fft, hop_length, win_length, normalized, center, onesided):
    return gems_istft(
        spectrogram=spec,
        length=length,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        normalized=normalized,
        center=center,
        pad_mode="reflect",
        onesided=onesided,
    )


@pytest.mark.inverse_spectrogram
@pytest.mark.parametrize("n_fft", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
def test_inverse_spectrogram_basic(n_fft, center, onesided):
    hop_length = n_fft // 4
    win_length = n_fft
    waveform = torch.randn(2, 16000, device=flag_gems.device, dtype=torch.float32)

    spec = _forward_spec(
        waveform, n_fft, hop_length, win_length, None, False, center, onesided
    )
    window = torch.hann_window(win_length, device=flag_gems.device)

    # torch.hann_window defaults to periodic=True, whose first sample is zero.
    # With center=False there is no symmetric padding, so the very first output
    # sample sees an overlap-add envelope of w[0]^2 = 0, violating torch.istft's
    # NOLA (nonzero overlap-add) condition. torchaudio raises RuntimeError in
    # this case; the Triton op mirrors that, so the comparison is not defined.
    if center is False:
        pytest.xfail(
            "NOLA condition violated for periodic Hann window with center=False "
            "(window[0] == 0 leads to a zero in the OLA envelope); see "
            "test_inverse_spectrogram_nola_violation for the symmetric check."
        )

    ref = _ref_istft(
        spec, None, window, n_fft, hop_length, win_length, False, center, onesided
    )
    out = _gems_istft(
        spec, None, window, n_fft, hop_length, win_length, False, center, onesided
    )

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    # IFFT + OLA reconstruction: larger n_fft accumulates more fp drift.
    atol = 1e-2 if n_fft >= 512 else 5e-3
    utils.gems_assert_close(out, ref, torch.float32, atol=atol, reduce_dim=1)


@pytest.mark.inverse_spectrogram
@pytest.mark.parametrize("n_fft", [256, 512])
@pytest.mark.parametrize("normalized", [True, False, "frame_length", "window"])
def test_inverse_spectrogram_normalized(n_fft, normalized):
    hop_length = n_fft // 4
    win_length = n_fft
    waveform = torch.randn(2, 8000, device=flag_gems.device, dtype=torch.float32)

    spec = _forward_spec(
        waveform, n_fft, hop_length, win_length, None, normalized, True, True
    )
    window = torch.hann_window(win_length, device=flag_gems.device)

    ref = _ref_istft(
        spec, None, window, n_fft, hop_length, win_length, normalized, True, True
    )
    out = _gems_istft(
        spec, None, window, n_fft, hop_length, win_length, normalized, True, True
    )

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    utils.gems_assert_close(out, ref, torch.float32, atol=1e-2, reduce_dim=1)


@pytest.mark.inverse_spectrogram
@pytest.mark.parametrize("n_fft", [256, 512])
@pytest.mark.parametrize("hop_length_factor", [2, 4])
def test_inverse_spectrogram_hop_length(n_fft, hop_length_factor):
    hop_length = n_fft // hop_length_factor
    win_length = n_fft
    waveform = torch.randn(1, 8000, device=flag_gems.device, dtype=torch.float32)

    spec = _forward_spec(
        waveform, n_fft, hop_length, win_length, None, False, True, True
    )
    window = torch.hann_window(win_length, device=flag_gems.device)

    ref = _ref_istft(
        spec, None, window, n_fft, hop_length, win_length, False, True, True
    )
    out = _gems_istft(
        spec, None, window, n_fft, hop_length, win_length, False, True, True
    )

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    utils.gems_assert_close(out, ref, torch.float32, atol=1e-2, reduce_dim=1)


@pytest.mark.inverse_spectrogram
@pytest.mark.parametrize("n_fft", [256, 512])
def test_inverse_spectrogram_win_length_less_than_n_fft(n_fft):
    hop_length = n_fft // 4
    win_length = n_fft // 2
    waveform = torch.randn(1, 8000, device=flag_gems.device, dtype=torch.float32)

    window = torch.hann_window(win_length, device=flag_gems.device)
    spec = torchaudio.functional.spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=None,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )

    ref = _ref_istft(
        spec, None, window, n_fft, hop_length, win_length, False, True, True
    )
    out = _gems_istft(
        spec, None, window, n_fft, hop_length, win_length, False, True, True
    )

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    utils.gems_assert_close(out, ref, torch.float32, atol=1e-2, reduce_dim=1)


@pytest.mark.inverse_spectrogram
@pytest.mark.parametrize("batch_shape", [(8000,), (3, 8000), (2, 3, 8000)])
def test_inverse_spectrogram_batch(batch_shape):
    n_fft = 256
    hop_length = n_fft // 4
    win_length = n_fft
    waveform = torch.randn(*batch_shape, device=flag_gems.device, dtype=torch.float32)

    window = torch.hann_window(win_length, device=flag_gems.device)
    spec = torchaudio.functional.spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=None,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )

    ref = _ref_istft(
        spec, None, window, n_fft, hop_length, win_length, False, True, True
    )
    out = _gems_istft(
        spec, None, window, n_fft, hop_length, win_length, False, True, True
    )

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    utils.gems_assert_close(out, ref, torch.float32, atol=1e-2, reduce_dim=1)


@pytest.mark.inverse_spectrogram
@pytest.mark.parametrize("n_fft", [256, 512])
def test_inverse_spectrogram_with_length(n_fft):
    hop_length = n_fft // 4
    win_length = n_fft
    waveform = torch.randn(1, 8001, device=flag_gems.device, dtype=torch.float32)

    window = torch.hann_window(win_length, device=flag_gems.device)
    spec = torchaudio.functional.spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=None,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )

    target_length = waveform.size(-1)
    ref = _ref_istft(
        spec, target_length, window, n_fft, hop_length, win_length, False, True, True
    )
    out = _gems_istft(
        spec, target_length, window, n_fft, hop_length, win_length, False, True, True
    )

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    utils.gems_assert_close(out, ref, torch.float32, atol=1e-2, reduce_dim=1)


@pytest.mark.inverse_spectrogram
def test_inverse_spectrogram_round_trip():
    """Reconstruction error after spec -> istft should be very small."""
    n_fft = 256
    hop_length = n_fft // 4
    win_length = n_fft
    waveform = torch.randn(1, 4000, device=flag_gems.device, dtype=torch.float32)
    window = torch.hann_window(win_length, device=flag_gems.device)

    spec = torchaudio.functional.spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=None,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )
    out = _gems_istft(
        spec, waveform.size(-1), window, n_fft, hop_length, win_length, False, True, True
    )
    # Trim reference region for fair comparison
    ref = waveform[..., : out.shape[-1]]
    utils.gems_assert_close(out, ref, torch.float32, atol=1e-1, reduce_dim=1)


@pytest.mark.inverse_spectrogram
@pytest.mark.parametrize("n_fft", [64, 256, 1024])
def test_inverse_spectrogram_nola_violation(n_fft):
    """center=False + periodic Hann window violates the NOLA condition.

    torch.hann_window(periodic=True) has window[0] == 0, and with center=False
    there is no analysis padding, so the first output sample's overlap-add
    envelope is exactly zero. torchaudio (via torch.istft) raises RuntimeError
    in this case; the Triton op must raise symmetrically rather than silently
    emit inf/nan.
    """
    hop_length = n_fft // 4
    win_length = n_fft
    waveform = torch.randn(
        2, 16000, device=flag_gems.device, dtype=torch.float32
    )
    window = torch.hann_window(win_length, device=flag_gems.device)
    spec = _forward_spec(
        waveform, n_fft, hop_length, win_length, None, False, False, True
    )

    # torchaudio reference: should raise.
    with pytest.raises(RuntimeError, match="window overlap add"):
        _ref_istft(
            spec, None, window, n_fft, hop_length, win_length, False, False, True
        )

    # Triton implementation: must raise the same error class symmetrically.
    with pytest.raises(RuntimeError, match="NOLA condition violated"):
        _gems_istft(
            spec, None, window, n_fft, hop_length, win_length, False, False, True
        )
