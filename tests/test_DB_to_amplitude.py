import torch
import torchaudio
import flag_audio


def test_db2amp_common():
    x = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float32).to("cuda")
    ref = 3.0
    power = 0.5
    output_tensor = flag_audio.DB_to_amplitude(x, ref, power)
    golden_tensor = torchaudio.functional.DB_to_amplitude(x, ref, power)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)

test_db2amp_common()