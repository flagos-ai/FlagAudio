[English|[中文版](./README_cn.md)]

## Introduction

FlagAudio is part of [FlagOS](https://flagos.io/).
FlagAudio is a multi-backend computing library that adheres to Audio standard interfaces. It delivers a high-performance computing solution designed for audio signal processing and speech AI applications, offering a complete processing chain from raw audio to model input.

FlagAudio is a high-performance general-purpose operator library implemented using the [Triton programming language](https://github.com/openai/triton) launched by OpenAI.

## Features

- Operators have undergone deep performance tuning
- Triton kernel call optimization
- Flexible multi-backend support mechanism

## Quick Installation
### Install Dependencies
```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```
### Install FlagAudio
```shell
git clone https://github.com/flagos-ai/FlagAudio.git
cd FlagAudio
pip install  .
```

On a test machine with PyTorch already installed, check the active interpreter and keep the existing accelerator build:

```shell
python -c "import sys, torch; print(sys.executable, torch.__version__)"
python -m pip install --no-deps -e .
```

The `torch>=2.6.0` runtime requirement does not reinstall a compatible existing package on every install. A normal `pip install .` may change Torch if the installed version does not satisfy the requirement. `--no-deps` preserves the preinstalled stack and assumes its other dependencies are present. `requirements-triton_env.txt` is a pinned CUDA setup example; the vendor extras in `pyproject.toml` are `ascend`, `iluvatar`, `mthreads`, and `nvidia` without SDK versions. Install extras with the matching vendor package index and compatible Torch and torchaudio versions.

## Batch tests and reports

`conf/operators.yaml` lists the operators. The scheduler runs accuracy tests in `tests/` and performance tests in `benchmark/`:

```shell
python tools/run_tests.py --list
python tools/run_tests.py --ops gain,contrast --gpus 0 --output-dir results_example
python tools/psum_text.py results_example/summary.json
python tools/psum_html.py results_example/summary.json
```

By default, all operators except those at the `removed` stage are selected. Use `--stages alpha,beta`, `--ops`, or `--op-list-file` to narrow selection. Each result directory contains pytest JSON reports and logs per operator plus `summary.json`; the report commands generate `summary.csv` and `summary.html`.

Multi-device scheduling uses `CUDA_VISIBLE_DEVICES` by default. On Ascend, set `BLAS_VENDOR=ascend` and pass `--device-env ASCEND_RT_VISIBLE_DEVICES`. Check that the preinstalled Torch, torchaudio, pytest, and PyYAML packages import successfully before installing this project with `--no-deps`.

Accuracy references default to the device on NVIDIA and other backends, and to CPU when `BLAS_VENDOR=iluvatar`. Override this with `--accuracy-reference cpu` or `--accuracy-reference device`. Performance timing always runs both implementations on the accelerator.

Performance tests use Triton's `kernel` timing mode by default: the torchaudio/PyTorch reference and FlagAudio implementation receive the same device inputs, and the report shows median GPU time and `reference time / FlagAudio time`. `--warmup-ms` and `--measurement-ms` specify durations in milliseconds. The older `--warmup` and `--iterations` names remain available but also mean milliseconds in this default mode, not invocation counts. A slow operator can get only one timed sample when its latency exceeds the measurement duration; increase `--measurement-ms` for those cases. A performance `PASS` means timing completed; use the accuracy phase to assess output correctness. Keep shapes, dtypes, PyTorch/torchaudio versions, and timing settings aligned when comparing different servers.

This project is licensed under the [Apache License (version 2.0)](./LICENSE).
