import torch
import torchaudio
import flag_audio


def test_gain_common():
    input_tensor = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32).to("cuda")
    gain_db = 6.0
    output_tensor = flag_audio.gain(input_tensor, gain_db)
    golden_tensor = torchaudio.functional.gain(input_tensor, gain_db)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)

test_gain_common()