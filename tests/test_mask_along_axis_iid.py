import torch
import torchaudio
import flag_audio


def mock_rand(size, **kwargs):
    return torch.ones(size, **kwargs) * 0.6

def test_mask_along_axis_iid_common():
    specgram = torch.rand(2, 2, 9, 4, device="cuda")
    mask_param = 3
    mask_value = 0.0
    axis = 3
    p = 1
    torch.rand = mock_rand
    output_triton = flag_audio.mask_along_axis_iid(
        specgram, mask_param, mask_value, axis, p
    )
    golden_output = torchaudio.functional.mask_along_axis_iid(
        specgram, mask_param, mask_value, axis, p
    )
    torch.testing.assert_close(output_triton, golden_output, rtol=1e-5, atol=1e-8)
    print(f"Output tensor: {output_triton}")
    print(f"Golden tensor: {golden_output}")

test_mask_along_axis_iid_common()