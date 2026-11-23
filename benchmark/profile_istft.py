"""Profile inverse_spectrogram to locate the real bottleneck."""

import time
import torch
import torchaudio

from flag_audio.ops.inverse_spectrogram import (
    inverse_spectrogram,
    _ola_window_kernel,
    _window_overlap_energy_kernel,
    _scale_kernel,
    _center_pad_window,
)


def timed(fn, *args, warmup=5, iters=50, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    device = "cuda"
    torch.manual_seed(0)
    n_fft = 1024
    batch = 16
    hop_length = n_fft // 4
    win_length = n_fft
    time_len = hop_length * 100 + n_fft

    waveform = torch.randn(batch, time_len, device=device, dtype=torch.float32)
    window = torch.hann_window(win_length, device=device, dtype=torch.float32)
    spec = torchaudio.functional.spectrogram(
        waveform=waveform, pad=0, window=window, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length, power=None,
        normalized=False, center=True, pad_mode="reflect", onesided=True,
    )

    kwargs = dict(length=None, pad=0, window=window, n_fft=n_fft,
                  hop_length=hop_length, win_length=win_length,
                  normalized=False, center=True, pad_mode="reflect", onesided=True)

    def torch_fn(s):
        return torchaudio.functional.inverse_spectrogram(spectrogram=s, **kwargs)

    def gems_fn(s):
        return inverse_spectrogram(spectrogram=s, **kwargs)

    t_torch = timed(torch_fn, spec)
    t_gems = timed(gems_fn, spec)
    print(f"[end-to-end]  torch={t_torch:.4f} ms  gems={t_gems:.4f} ms  ratio={t_torch/t_gems:.3f}")

    # ---- Now break down the Gems pipeline stage by stage ----
    import triton

    n_freq_in = spec.size(-2)
    n_frames = spec.size(-1)
    spec_3d = spec.reshape(-1, n_freq_in, n_frames).to(torch.complex64)
    b = spec_3d.shape[0]

    # Stage 1: Hermitian reconstruction
    def stage1_hermitian():
        full_real = torch.empty((b, n_fft, n_frames), device=device, dtype=torch.float32)
        full_imag = torch.empty_like(full_real)
        full_real[:, 0, :] = spec_3d[:, 0, :].real
        full_imag[:, 0, :] = spec_3d[:, 0, :].imag
        full_real[:, n_fft // 2, :] = spec_3d[:, n_freq_in - 1, :].real
        full_imag[:, n_fft // 2, :] = spec_3d[:, n_freq_in - 1, :].imag
        pos_idx = torch.arange(1, n_fft // 2, device=device)
        pos_real = spec_3d[:, pos_idx, :].real
        pos_imag = spec_3d[:, pos_idx, :].imag
        full_real[:, pos_idx, :] = pos_real
        full_imag[:, pos_idx, :] = pos_imag
        neg_idx = n_fft - pos_idx
        full_real[:, neg_idx, :] = pos_real
        full_imag[:, neg_idx, :] = -pos_imag
        return full_real, full_imag

    # Stage 2: layout permute + contiguous + complex
    def stage2_layout(full_real, full_imag):
        total_frames = b * n_frames
        fr = full_real.permute(0, 2, 1).reshape(total_frames, n_fft).contiguous()
        fi = full_imag.permute(0, 2, 1).reshape(total_frames, n_fft).contiguous()
        return torch.complex(fr, fi)

    # Stage 3: IFFT
    def stage3_ifft(full_complex):
        ift = torch.fft.ifft(full_complex, n=n_fft, dim=-1)
        return ift.real.to(torch.float32).contiguous()

    # Stage 4: OLA
    def stage4_ola(full_real):
        window_synth = _center_pad_window(window, win_length, n_fft)
        target_len = n_fft + hop_length * (n_frames - 1)
        out_waveform = torch.zeros((b, target_len), device=device, dtype=torch.float32)
        norm_envelope = torch.zeros_like(out_waveform)
        block_nfft = 1
        while block_nfft < n_fft:
            block_nfft *= 2
        grid = (b * n_frames,)
        _ola_window_kernel[grid](
            full_real, window_synth, out_waveform, n_fft, n_frames * n_fft,
            target_len, n_fft, hop_length, target_len, n_frames,
            BLOCK_NFFT=block_nfft, num_warps=4, num_stages=1,
        )
        _window_overlap_energy_kernel[grid](
            window_synth, norm_envelope, target_len, hop_length, n_fft,
            target_len, n_frames, BLOCK_NFFT=block_nfft, num_warps=4, num_stages=1,
        )
        return out_waveform, norm_envelope

    # Stage 5: NOLA check + scale
    def stage5_nola_scale(out_waveform, norm_envelope):
        # NOLA check (the .any() forces a sync)
        env_slice = norm_envelope
        _ = torch.le(env_slice, torch.zeros((), device=device)).any()
        BLOCK = 256
        grid_norm = (triton.cdiv(out_waveform.numel(), BLOCK),)
        _scale_kernel[grid_norm](out_waveform, norm_envelope, out_waveform.numel(), BLOCK_SIZE=BLOCK)
        return out_waveform

    # Warm up & time each stage independently (including data needed downstream)
    fr, fi = stage1_hermitian()
    fc = stage2_layout(fr, fi)
    ift_real = stage3_ifft(fc)
    out, norm = stage4_ola(ift_real)
    _ = stage5_nola_scale(out.clone(), norm.clone())

    t1 = timed(stage1_hermitian)
    print(f"[stage1 hermitian]   {t1:.4f} ms")

    fr, fi = stage1_hermitian()
    t2 = timed(stage2_layout, fr, fi)
    print(f"[stage2 layout]      {t2:.4f} ms")

    fc = stage2_layout(fr, fi)
    t3 = timed(stage3_ifft, fc)
    print(f"[stage3 ifft]        {t3:.4f} ms")

    ift_real = stage3_ifft(fc)
    t4 = timed(stage4_ola, ift_real)
    print(f"[stage4 ola]         {t4:.4f} ms")

    out, norm = stage4_ola(ift_real)
    # clone to ensure prev OLA doesn't alias the buffers we time
    t5 = timed(stage5_nola_scale, out.clone(), norm.clone())
    print(f"[stage5 nola+scale]  {t5:.4f} ms")

    # Also time just torch.fft.ifft on the exact same input tensor to isolate cuFFT
    def raw_ifft():
        return torch.fft.ifft(fc, n=n_fft, dim=-1)
    t_ifft_only = timed(raw_ifft)
    print(f"[ifft only (cuFFT)]  {t_ifft_only:.4f} ms")


if __name__ == "__main__":
    main()
