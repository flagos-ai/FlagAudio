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

"""Triton implementation of torchaudio.functional.inverse_spectrogram.

The full inverse pipeline (Hermitian expansion, IFFT, synthesis window,
overlap-add, normalization undo) is fused into **two** Triton kernels:

  * ``_istfft_ola_kernel`` — per frame: rebuilds the full ``n_fft`` complex
    spectrum from a one-sided input via Hermitian symmetry, performs an
    in-register radix-4 IFFT (DIT, twiddles conjugated), applies the
    synthesis window, and atomically overlap-adds into the output waveform.
  * ``_window_energy_kernel`` — accumulates ``window^2`` across frames into
    the normalization envelope (kept separate: scale must wait for all OLA
    contributions, which a single kernel cannot synchronize).

Division by the envelope replaces the former ``_scale_kernel`` and is done
with one fused ``torch.where`` pointwise op (not a Triton kernel). The IFFT
reuses the flag_gems FFT lookup tables (``_bitrev_indices`` /
``_twiddle_tables``) — the same tables as the forward path, with the twiddle
imaginary part negated inside the kernel.
"""

from typing import Optional, Union

import torch
import triton
import triton.language as tl

from flag_gems.ops.fft import _bitrev_indices, _twiddle_tables, _is_power_of_two, _log2


# -----------------------------------------------------------------------------
# Fused kernel A: Hermitian expand + radix-4 IFFT + window + overlap-add
# -----------------------------------------------------------------------------
@triton.jit
def _istfft_ola_kernel(
    spec_real_ptr,          # (batch, n_freq, n_frames) float32
    spec_imag_ptr,          # (batch, n_freq, n_frames) float32
    window_ptr,             # (n_fft,) center-padded synthesis window
    bitrev_ptr,             # (n_fft,) bit-reversal index table
    tw_real_ptr,            # (n_fft//2 - 1,) real twiddles
    tw_imag_ptr,            # (n_fft//2 - 1,) imag twiddles (forward sign)
    out_ptr,                # (batch, target_len) float32
    stride_bin,             # elements between adjacent bins of a frame (== n_frames)
    stride_batch,           # elements between batches in spec buffer (== n_freq * n_frames)
    out_stride_batch,       # elements between batches in the output waveform (== target_len)
    hop_length,
    signal_len,
    n_frames,
    scale,                  # float: (1/n_fft) * forward-normalization-undo factor
    ONE_SIDED: tl.constexpr,
    NFFT: tl.constexpr,
    LOG_N: tl.constexpr,
    N_RADIX4: tl.constexpr,
    FIRST_STAGE: tl.constexpr,
):
    """Hermitian expand + radix-4 IFFT + synthesis window + atomic OLA."""
    pid = tl.program_id(0)
    frame_idx = pid % n_frames
    batch_idx = pid // n_frames

    offs = tl.arange(0, NFFT)
    mask = offs < NFFT

    # ---- 1) Hermitian expansion in registers -------------------------------
    # For ONE_SIDED input (n_freq = NFFT//2 + 1), bins NFFT//2+1 .. NFFT-1 are
    # filled by mirroring: X[N-k] = conj(X[k]). Map each output bin to its
    # source index in the one-sided buffer and tag conjugate bins.
    nyq = NFFT // 2
    neg_mask = (offs > nyq) & (offs < NFFT)
    src_idx = tl.where(ONE_SIDED & neg_mask, NFFT - offs, offs)
    conj_mask = ONE_SIDED & neg_mask

    frame_base = batch_idx * stride_batch + frame_idx
    xr = tl.load(
        spec_real_ptr + frame_base + src_idx * stride_bin, mask=mask, other=0.0
    )
    xi = tl.load(
        spec_imag_ptr + frame_base + src_idx * stride_bin, mask=mask, other=0.0
    )
    xi = tl.where(conj_mask, -xi, xi)

    # ---- 2) IFFT via the conjugation identity ifft(X) = conj(fft(conj(X))) / N
    # ----    Reuses the forward radix-4 DIT structure unchanged; the only
    #        IFFT-specific tweaks are: flip the input's imaginary sign (conj),
    #        flip the output's imaginary sign (conj again), and apply 1/N
    #        through the scalar ``scale`` below. This mirrors what the prior
    #        host-side ``torch.conj(flag_gems.fft(torch.conj(...)))/n_fft`` did
    #        but keeps the whole FFT in-kernel so it can be fused with the OLA.
    real = xr
    imag = -xi                              # conj(X)
    rev = tl.load(bitrev_ptr + offs, mask=mask, other=0)
    real = tl.gather(real, rev, axis=0)
    imag = tl.gather(imag, rev, axis=0)

    # ---- 3) Radix-2 first stage (only when LOG_N is odd) -------------------
    if LOG_N % 2 == 1:
        idx = offs
        even_idx = idx & ~1
        odd_idx = even_idx | 1

        u_real = tl.gather(real, even_idx, axis=0)
        u_imag = tl.gather(imag, even_idx, axis=0)
        v_real = tl.gather(real, odd_idx, axis=0)
        v_imag = tl.gather(imag, odd_idx, axis=0)

        tw_real = tl.load(tw_real_ptr + 0)
        tw_imag = tl.load(tw_imag_ptr + 0)

        v_tw_real = v_real * tw_real - v_imag * tw_imag
        v_tw_imag = v_real * tw_imag + v_imag * tw_real

        is_even = (idx & 1) == 0
        real = tl.where(is_even, u_real + v_tw_real, u_real - v_tw_real)
        imag = tl.where(is_even, u_imag + v_tw_imag, u_imag - v_tw_imag)

    # ---- 4) Radix-4 stages (forward twiddles; conj(X) input -> fft(conj(X)))
    for r4 in tl.static_range(N_RADIX4):
        stage_s = FIRST_STAGE + r4 * 2
        m = 1 << (stage_s + 1)
        quarter = m >> 2
        half = m >> 1

        idx = offs
        pos = idx & (m - 1)
        j = pos & (quarter - 1)
        base = idx - pos
        i0 = base + j
        i1 = i0 + quarter
        i2 = i1 + quarter
        i3 = i2 + quarter

        x0_real = tl.gather(real, i0, axis=0)
        x0_imag = tl.gather(imag, i0, axis=0)
        x1_real = tl.gather(real, i1, axis=0)
        x1_imag = tl.gather(imag, i1, axis=0)
        x2_real = tl.gather(real, i2, axis=0)
        x2_imag = tl.gather(imag, i2, axis=0)
        x3_real = tl.gather(real, i3, axis=0)
        x3_imag = tl.gather(imag, i3, axis=0)

        base_tw1 = (1 << (stage_s - 1)) - 1
        base_tw2 = (1 << stage_s) - 1
        tw1_idx = base_tw1 + j
        tw2_idx = base_tw2 + j
        tw1_real = tl.load(tw_real_ptr + tw1_idx, mask=offs < NFFT, other=1.0)
        tw1_imag = tl.load(tw_imag_ptr + tw1_idx, mask=offs < NFFT, other=0.0)
        tw2_real = tl.load(tw_real_ptr + tw2_idx, mask=offs < NFFT, other=1.0)
        tw2_imag = tl.load(tw_imag_ptr + tw2_idx, mask=offs < NFFT, other=0.0)

        # Twiddle multiplication
        t1_real = x1_real * tw1_real - x1_imag * tw1_imag
        t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
        t3_real = x3_real * tw1_real - x3_imag * tw1_imag
        t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

        # Butterfly
        u0_real = x0_real + t1_real
        u0_imag = x0_imag + t1_imag
        u1_real = x0_real - t1_real
        u1_imag = x0_imag - t1_imag
        v0_real = x2_real + t3_real
        v0_imag = x2_imag + t3_imag
        v1_real = x2_real - t3_real
        v1_imag = x2_imag - t3_imag

        v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
        v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
        # w3 = tw2 * (-j) = (tw2_imag, -tw2_real)
        v1_tw_real = v1_real * tw2_imag - v1_imag * (-tw2_real)
        v1_tw_imag = v1_real * (-tw2_real) + v1_imag * tw2_imag

        o0_real = u0_real + v0_tw_real
        o0_imag = u0_imag + v0_tw_imag
        o2_real = u0_real - v0_tw_real
        o2_imag = u0_imag - v0_tw_imag
        o1_real = u1_real + v1_tw_real
        o1_imag = u1_imag + v1_tw_imag
        o3_real = u1_real - v1_tw_real
        o3_imag = u1_imag - v1_tw_imag

        three_quarter = quarter + half
        m0 = pos < quarter
        m1 = (pos >= quarter) & (pos < half)
        m2 = (pos >= half) & (pos < three_quarter)
        real = tl.where(m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real)))
        imag = tl.where(m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag)))

    # ---- 4b) conj(fft(conj(X)))/N -> take real part (Hermitian input -> real)
    # For a Hermitian-symmetric spectrum the true ifft result is purely real,
    # so we only need real (the imag column is numerically ~0 after the conj
    # trick). The 1/N scaling and any forward-normalization undo are folded
    # into the scalar ``scale``.
    real = real * scale

    # ---- 5) Synthesis window + overlap-add --------------------------------
    win = tl.load(window_ptr + offs, mask=mask, other=0.0)
    sample = real * win

    sig_idx = frame_idx * hop_length + offs
    sig_mask = mask & (sig_idx < signal_len)
    tl.atomic_add(
        out_ptr + batch_idx * out_stride_batch + sig_idx, sample, mask=sig_mask
    )


# -----------------------------------------------------------------------------
# Fused kernel B: window-energy OLA (norm envelope)
# -----------------------------------------------------------------------------
@triton.jit
def _window_energy_kernel(
    window_ptr,
    norm_ptr,
    norm_stride_batch,
    hop_length,
    n_fft,
    signal_len,
    n_frames,
    BLOCK_NFFT: tl.constexpr,
):
    """Accumulate window^2 over all frames into the OLA normalization envelope.

    Kept as a separate kernel: scale must wait for every frame's OLA
    contribution, which cannot be synchronized inside a single Triton kernel.
    """
    pid = tl.program_id(0)
    frame_idx = pid % n_frames
    batch_idx = pid // n_frames

    offs = tl.arange(0, BLOCK_NFFT)
    mask = offs < n_fft

    win = tl.load(window_ptr + offs, mask=mask, other=0.0)
    contrib = win * win

    sig_idx = frame_idx * hop_length + offs
    sig_mask = mask & (sig_idx < signal_len)
    tl.atomic_add(
        norm_ptr + batch_idx * norm_stride_batch + sig_idx, contrib, mask=sig_mask
    )


def _center_pad_window(window: torch.Tensor, win_length: int, n_fft: int) -> torch.Tensor:
    """Return the window centered inside an n_fft-length tensor (matches torch.stft)."""
    w = window.to(torch.float32).contiguous()
    if w.numel() < n_fft:
        left = (n_fft - w.numel()) // 2
        right = n_fft - w.numel() - left
        w = torch.nn.functional.pad(w, (left, right))
    return w


def inverse_spectrogram(
    spectrogram: torch.Tensor,
    length: Optional[int],
    pad: int,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    normalized: Union[bool, str],
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
) -> torch.Tensor:
    """Triton-based inverse spectrogram via overlap-add.

    Args:
        spectrogram: Complex tensor of shape (..., freq, time).
        length: Desired output waveform length (trim or pad to match).
        pad: Two-sided padding associated with the forward transform.
        window: Window tensor used for the forward transform.
        n_fft: FFT size (must be a positive power of two <= 1024; the fused
            radix-4 IFFT kernel uses flag_gems' lookup tables which are
            sized for this range).
        hop_length: Hop length between frames.
        win_length: Size of the window before centering.
        normalized: Normalization used during the forward STFT.
        center: Whether the forward transform used center padding.
        pad_mode: Kept for API compatibility (unused, matches torchaudio).
        onesided: Whether the input spectrum is one-sided.

    Returns:
        Waveform tensor of shape (..., time).
    """
    if not _is_power_of_two(n_fft) or n_fft > 1024:
        raise ValueError(
            f"n_fft={n_fft} must be a positive power of two and <= 1024"
        )
    if not spectrogram.is_complex():
        raise TypeError("inverse_spectrogram expects a complex-valued spectrogram")

    device = spectrogram.device
    original_shape = spectrogram.size()
    n_freq_in = original_shape[-2]
    n_frames = original_shape[-1]
    batch_dims = original_shape[:-2]

    # Sanity-check the input frequency axis against n_fft.
    if onesided and n_freq_in != n_fft // 2 + 1:
        raise ValueError(
            f"onesided spectrum expects freq={n_fft // 2 + 1}, got {n_freq_in}"
        )
    if not onesided and n_freq_in != n_fft:
        raise ValueError(
            f"two-sided spectrum expects freq={n_fft}, got {n_freq_in}"
        )

    # Flatten leading dims into a single batch; split real/imag for the kernel.
    spec_3d = spectrogram.reshape(-1, n_freq_in, n_frames).to(torch.complex64)
    batch = spec_3d.shape[0]
    spec_real = spec_3d.real.contiguous()
    spec_imag = spec_3d.imag.contiguous()

    # ------------------------------------------------------------------
    # 1) Synthesis window + output buffers + FFT lookup tables
    # ------------------------------------------------------------------
    window_synth = _center_pad_window(window, win_length, n_fft)
    target_len = n_fft + hop_length * (n_frames - 1)
    out_waveform = torch.zeros(
        (batch, target_len), device=device, dtype=torch.float32
    )
    norm_envelope = torch.zeros_like(out_waveform)

    # ---- Scalar scale: fuses (1/N) with the forward-normalization undo ----
    # The forward STFT divides the spectrum by sqrt(sum(window^2)) or sqrt(N);
    # the inverse must multiply it back. Folding these into a single scalar
    # avoids two eager elementwise passes on the full complex spectrum.
    scale = 1.0 / n_fft
    if normalized is True or normalized == "window":
        win_orig = window.to(torch.float32)
        if win_orig.numel() != win_length:
            win_orig = (
                win_orig[:win_length] if win_orig.numel() > win_length else win_orig
            )
        scale = scale * win_orig.pow(2.0).sum().sqrt().item()
    elif normalized == "frame_length":
        scale = scale * (n_fft ** 0.5)

    # ---- FFT lookup tables (shared with the forward spectrogram kernel) ----
    # bitrev is direction-agnostic; the twiddle imag sign is flipped inside
    # the kernel to turn the forward DIT into an inverse DIT.
    bitrev = _bitrev_indices(n_fft, device)
    tw_real, tw_imag = _twiddle_tables(n_fft, device)
    log_n = _log2(n_fft)
    if log_n % 2 == 1:
        n_radix4 = (log_n - 1) // 2
        first_stage = 2
    else:
        n_radix4 = log_n // 2
        first_stage = 1

    # ------------------------------------------------------------------
    # 2) Launch the two fused kernels (single grid over batch * n_frames)
    # ------------------------------------------------------------------
    # Kernel A: Hermitian expand + radix-4 IFFT + synthesis window + atomic
    # OLA into out_waveform. Kernel B: window^2 OLA into norm_envelope.
    # Running them back-to-back is safe: they write to distinct buffers, so
    # there is no read-after-write hazard, and the division that consumes
    # both happens later on the host.
    grid = (batch * n_frames,)
    _istfft_ola_kernel[grid](
        spec_real,
        spec_imag,
        window_synth,
        bitrev,
        tw_real,
        tw_imag,
        out_waveform,
        n_frames,              # stride_bin (adjacent bin of a single frame)
        n_freq_in * n_frames,  # stride_batch
        target_len,            # out_stride_batch
        hop_length,
        target_len,            # signal_len
        n_frames,
        scale,
        ONE_SIDED=onesided,
        NFFT=n_fft,
        LOG_N=log_n,
        N_RADIX4=n_radix4,
        FIRST_STAGE=first_stage,
        num_warps=4,
        num_stages=1,
    )
    _window_energy_kernel[grid](
        window_synth,
        norm_envelope,
        target_len,            # norm_stride_batch
        hop_length,
        n_fft,
        target_len,
        n_frames,
        BLOCK_NFFT=n_fft,
        num_warps=4,
        num_stages=1,
    )

    # ------------------------------------------------------------------
    # 3) NOLA check + normalize by the window-energy envelope
    # ------------------------------------------------------------------
    # Mirror torch.istft's NOLA (nonzero overlap-add) condition: the window-
    # energy envelope must be strictly positive at every output sample,
    # otherwise the least-squares estimate is undefined (division by zero).
    # torchaudio raises RuntimeError in this case; we do the same so the two
    # paths behave symmetrically. Only the slice that survives the final
    # trim is checked, since samples outside the output region are discarded
    # and irrelevant.
    if center and length is not None:
        nola_start, nola_end = n_fft // 2, n_fft // 2 + length
    elif center:
        nola_start, nola_end = n_fft // 2, n_fft // 2 + hop_length * (n_frames - 1)
    elif length is not None:
        nola_start, nola_end = 0, length
    else:
        nola_start, nola_end = 0, target_len

    nola_end = min(nola_end, target_len)
    if nola_end > nola_start:
        env_slice = norm_envelope[:, nola_start:nola_end]
        # torch.istft checks window_overlap_add.min() == 0; this comparison
        # matches that exactly (no tolerance band) so the Triton op and the
        # torch reference raise on the same inputs.
        if torch.le(env_slice, torch.zeros((), device=device, dtype=torch.float32)).any():
            raise RuntimeError(
                "inverse_spectrogram: NOLA condition violated (window overlap add "
                f"min is zero) for n_fft={n_fft}, hop_length={hop_length}, "
                f"win_length={win_length}, center={center}. The window-energy "
                "envelope has a zero within the output region; choose a window / "
                "hop pair that satisfies nonzero overlap (e.g. center=True, or a "
                "non-vanishing window such as torch.ones)."
            )

    # Divide by the OLA envelope, guarding against zeros outside the checked
    # region (which are not NOLA violations, just trimmed-away samples).
    out_waveform = torch.where(
        norm_envelope > 0,
        out_waveform / norm_envelope.clamp_min(1e-12),
        out_waveform,
    )

    # ------------------------------------------------------------------
    # 6) Trim center-padding offset and match requested length
    # ------------------------------------------------------------------
    if center and length is not None:
        # Keep [n_fft//2, n_fft//2 + length), zero-padding the tail when the
        # OLA buffer does not reach that far. Matches torch.istft(center=True).
        start = n_fft // 2
        end = start + length
        if end > out_waveform.size(-1):
            pad = end - out_waveform.size(-1)
            out_waveform = torch.nn.functional.pad(out_waveform, (0, pad))
        out_waveform = out_waveform[..., start:end]
    elif center:
        start = n_fft // 2
        end = start + hop_length * (n_frames - 1)
        out_waveform = out_waveform[:, start:end]
    elif length is not None:
        if length > out_waveform.size(-1):
            out_waveform = torch.nn.functional.pad(
                out_waveform, (0, length - out_waveform.size(-1))
            )
        else:
            out_waveform = out_waveform[..., :length]

    out_waveform = out_waveform.reshape(batch_dims + (-1,))
    return out_waveform


# -----------------------------------------------------------------------------
# Smoke test (requires a device and torchaudio)
# -----------------------------------------------------------------------------
def test_op():
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    n_fft = 256
    win_length = n_fft
    hop_length = n_fft // 4
    waveform = torch.randn(2, 8001, device=device, dtype=torch.float32)
    window = torch.hann_window(win_length, device=device)

    spec = torchaudio.functional.spectrogram(
        waveform,
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

    ref = torchaudio.functional.inverse_spectrogram(
        spec,
        length=waveform.size(-1),
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
    out = inverse_spectrogram(
        spec,
        length=waveform.size(-1),
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

    assert ref.shape == out.shape, f"shape mismatch: {ref.shape} vs {out.shape}"
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("test_op passed:", out.shape)


if __name__ == "__main__":
    test_op()
