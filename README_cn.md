<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

[中文版|[English](./README.md)]

## 介绍

FlagAudio 是 [FlagOS](https://flagos.io/) 的一部分。
FlagAudio是一个遵循Audio标准的接口的面向多种芯片后端的计算库，它定义了音频信号处理和语音AI应用设计的高性能计算库，提供从原始音频到模型输入的完整处理链路。

FlagAudio 是一个使用 OpenAI 推出的[Triton 编程语言](https://github.com/openai/triton)实现的高性能通用算子库，

## 特性

- 算子已经过深度性能调优
- Triton kernel 调用优化
- 灵活的多后端支持机制


## 快速安装
### 安装依赖
```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```
### 安装FlagAudio
```shell
git clone https://github.com/flagos-ai/FlagAudio.git
cd FlagAudio
pip install  .
```

已有预装 PyTorch 的测试环境，先检查当前解释器与版本：

```shell
python -c "import sys, torch; print(sys.executable, torch.__version__)"
python -m pip install --no-deps -e .
```

`torch>=2.6.0` 是运行时依赖声明，并不表示每次都会重新安装。常规 `pip install .` 会检查已安装版本；版本不满足时可能更换 PyTorch 及其设备依赖。上面的 `--no-deps` 会保留预装环境，但要求环境已具备本项目所需依赖。`requirements-triton_env.txt` 是固定 CUDA 版本的从零部署示例；`pyproject.toml` 的厂商 extra 使用 `ascend`、`iluvatar`、`mthreads`、`nvidia`，不指定 SDK 版本。安装 extra 时仍应使用对应厂商的软件源，并确认 Torch 与 torchaudio 等组件版本匹配。

## 批量测试与报告

算子清单在 `conf/operators.yaml`。精度测试运行 `tests/`，性能测试运行 `benchmark/`：

```shell
python tools/run_tests.py --list
python tools/run_tests.py --ops gain,contrast --gpus 0 --output-dir results_example
python tools/psum_text.py results_example/summary.json
python tools/psum_html.py results_example/summary.json
```

默认测试清单中的全部非 `removed` 算子；可用 `--stages alpha,beta` 按阶段筛选，也可用 `--ops` 或 `--op-list-file` 指定算子。结果目录包含每个算子的 pytest JSON、日志和总 `summary.json`，两个汇总命令分别生成 `summary.csv` 与 `summary.html`。

多卡调度默认通过 `CUDA_VISIBLE_DEVICES` 指定设备；昇腾服务器使用 `--device-env ASCEND_RT_VISIBLE_DEVICES`，并在运行前设置 `BLAS_VENDOR=ascend`。预装环境应先确认 Torch、torchaudio、pytest、PyYAML 等依赖可导入，再按上面的 `--no-deps` 方式安装本项目。

精度测试的参考设备默认自动选择：`BLAS_VENDOR=iluvatar` 时使用 CPU，其他后端使用设备本身；可用 `--accuracy-reference cpu` 或 `--accuracy-reference device` 显式指定。此选项不影响性能测试，性能基线和 FlagAudio 算子都在加速卡上计时。

性能测试默认使用 Triton 的 `kernel` 计时模式：同一组设备输入分别运行 torchaudio/PyTorch 参考实现和 FlagAudio 实现，报告 GPU 耗时的中位数及 `参考耗时 / FlagAudio 耗时`。`--warmup-ms` 和 `--measurement-ms` 设置预热与采样时长（毫秒）；旧参数 `--warmup` 和 `--iterations` 仍可使用，但在默认模式下也表示毫秒，而非执行次数。单次耗时超过采样时长的大算子可能只有一次有效采样，应适当提高 `--measurement-ms`。性能阶段的 `PASS` 只表示计时成功；数值正确性以同次运行的精度测试结果为准。比较不同服务器的结果时，应对齐输入形状、dtype、PyTorch/torchaudio 版本和计时参数。




本项目采用 [Apache License (version 2.0)](./LICENSE) 授权许可。
