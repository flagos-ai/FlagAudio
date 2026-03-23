import torch
import torchaudio
import flag_audio

def mock_rand(size, **kwargs):
    return torch.ones(size, **kwargs) * 0.6

def test_mask_along_axis_common():
    specgram = torch.rand(2, 2, 4, 4, device="cuda")
    mask_param = 6
    mask_value = 0.0
    axis = 2
    p = 0.5
    torch.rand = mock_rand
    output_triton = flag_audio.mask_along_axis(specgram, mask_param, mask_value, axis, p)
    golden_output = torchaudio.functional.mask_along_axis(
        specgram, mask_param, mask_value, axis, p
    )
    torch.testing.assert_close(output_triton, golden_output, rtol=1e-5, atol=1e-8)
    print(output_triton, golden_output)

test_mask_along_axis_common()