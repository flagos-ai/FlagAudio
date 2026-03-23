import torch
import torchaudio
import flag_audio

def test_preemphasis_common():
    waveform = torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=torch.float32).to(
        "cuda"
    )
    coeff = 0.97
    output_tensor = flag_audio.preemphasis(waveform, coeff)
    golden_tensor = torchaudio.functional.preemphasis(waveform, coeff)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)

test_preemphasis_common()