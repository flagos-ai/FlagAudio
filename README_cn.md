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




本项目采用 [Apache License (version 2.0)](./LICENSE) 授权许可。
