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

import pytest
import torch

import flag_audio
import torchaudio

from .accuracy_utils import gems_assert_close, to_reference


@pytest.mark.add_noise
@pytest.mark.parametrize(
    "shape",
    [
        (8,),          # 1D：单条 + 标量 snr
        (2, 8),        # 2D：多条
        (256,),        # L 恰好 = BLOCK_SIZE，边界
        (257,),        # L = BLOCK_SIZE+1，第二段只有 1 个点
        (3, 2, 1000),  # 多维 + 非整除多段
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length_ratio", [None, 1.0, 0.5])
def test_accuracy_add_noise(shape, dtype, length_ratio):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    noise = torch.rand(shape, dtype=dtype, device=flag_audio.device)
    snr = torch.rand(shape[:-1], dtype=dtype, device=flag_audio.device)
    length = (
        torch.ones_like(snr, dtype=torch.int64, device=flag_audio.device)
        * int(shape[-1] * length_ratio)
        if length_ratio is not None
        else None
    )
    ref_inp1_tensor = to_reference(input_tensor, False)
    ref_noise_tensor = to_reference(noise, False)
    ref_snr_tensor = to_reference(snr, False)
    ref_length_tensor = to_reference(length, False) if length_ratio is not None else None

    ref_out = torchaudio.functional.add_noise(ref_inp1_tensor, ref_noise_tensor, ref_snr_tensor, lengths=ref_length_tensor)
    res_out = flag_audio.ops.add_noise(input_tensor, noise, snr, lengths=length)

    # half 精度下 scale 的舍入路径与 torchaudio 不同（我们全程 fp32），
    # 差异实测仅 1~3 ULP，按 dtype 放宽绝对容差
    atol = {
        torch.float32: 1e-4,
        torch.float16: 5e-3,
        torch.bfloat16: 4e-2,
    }[dtype]
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.add_noise
def test_add_noise_invalid_shapes():
    w = torch.randn(2, 8, device=flag_audio.device)
    n = torch.rand(2, 8, device=flag_audio.device)
    # snr 维度数不匹配：应为 (2,) 而不是标量
    with pytest.raises(ValueError):
        flag_audio.ops.add_noise(w, n, torch.tensor(0.5, device=flag_audio.device))
    # 最后一维长度不一致
    with pytest.raises(ValueError):
        flag_audio.ops.add_noise(
            w,
            torch.rand(2, 7, device=flag_audio.device),
            torch.rand(2, device=flag_audio.device),
        )
