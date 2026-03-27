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

This project is licensed under the [Apache License (version 2.0)](./LICENSE).
