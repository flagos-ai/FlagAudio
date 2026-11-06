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

"""Benchmark for the Triton ``inverse_spectrogram`` op against torchaudio."""

import pytest
import torch
import torchaudio

from flag_audio.ops.inverse_spectrogram import inverse_spectrogram as gems_istft

from .attri_util import DEFAULT_METRICS
from .performance_utils import Benchmark


class InverseSpectrogramBenchmark(Benchmark):
    """Benchmark for inverse_spectrogram operation."""

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_fft_values = [256, 512, 1024]
        self.batch_sizes = [1, 4, 16]
        self.hop_factors = [4]  # hop_length = n_fft // hop_factor

    def get_input_iter(self, dtype):
        for n_fft in self.n_fft_values:
            for batch in self.batch_sizes:
                for hop_factor in self.hop_factors:
                    hop_length = n_fft // hop_factor
                    win_length = n_fft
                    time_len = hop_length * 100 + n_fft
                    waveform = torch.randn(
                        batch, time_len, device=self.device, dtype=torch.float32
                    )
                    window = torch.hann_window(
                        win_length, device=self.device, dtype=torch.float32
                    )
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
                    yield (spec.contiguous(), window, n_fft, hop_length, win_length)

    def get_tflops(self, op, *args, **kwargs):
        spec, window, n_fft, hop_length, win_length = args
        batch = spec.shape[0]
        n_frames = spec.shape[-1]
        # IFFT is O(N log N) per frame; 5 flops per radix-2 butterfly.
        flops_per_ifft = 5 * n_fft * (n_fft.bit_length() - 1)
        # Plus OLA windowing + accumulation: ~3 flops per sample.
        flops_per_ola = 3 * n_fft
        return batch * n_frames * (flops_per_ifft + flops_per_ola)


@pytest.mark.inverse_spectrogram
def test_perf_inverse_spectrogram():
    def torch_istft(spec, window, n_fft, hop_length, win_length):
        return torchaudio.functional.inverse_spectrogram(
            spectrogram=spec,
            length=None,
            pad=0,
            window=window,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )

    def gems_istft_fn(spec, window, n_fft, hop_length, win_length):
        return gems_istft(
            spectrogram=spec,
            length=None,
            pad=0,
            window=window,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )

    bench = InverseSpectrogramBenchmark(
        op_name="inverse_spectrogram",
        torch_op=torch_istft,
        dtypes=[torch.float32],
    )
    bench.set_gems(gems_istft_fn)
    bench.run()
