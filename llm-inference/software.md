---
title: "Appendix: Inference Runtimes"
weight: 102
description: "LLM Serving Stacks, TensorRT, Triton, vLLM, and SGLang"
---

LLMs (or any other type of model) are not executed as a single GPU kernel.
Instead, different stages of the LLM are executed as one or more kernels;
examples of common kernels are operations like matrix multiplication, sigmoid
and ReLU, and attention kernels. Deploying a specific model for inference
requires specific kernels to be composed together, which is managed by an
execution runtime.

NVIDIA’s general-purpose inference runtime is
[TensorRT](https://developer.nvidia.com/tensorrt). TensorRT has both a compiler
and a runtime: when a model is to be deployed there is a one-time compilation
phase, which analyses the execution graph of the model, selects the appropriate
kernels, and optimizes for the specific GPU using techniques such as kernel
fusion to reduce the back and forth of data and execution between the GPU and
the host memory/CPU. TensorRT pre-dates LLMs and did not include attention
kernels; these were developed as part of
[FasterTransformer](https://github.com/NVIDIA/FasterTransformer), which is now
deprecated and rolled back into TensorRT as TensorRT-LLM.

While TensorRT has a network-facing component, this is primarily targeted for
development; production deployments typically use the [Triton Inference
Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html).
This splits the inference stack into a backend, which is responsible for
actually running the model on GPUs (TensorRT), and a frontend which is
responsible for routing and batching requests to the appropriate backends.

The cross-platform serving stack is largely serviced by
[vLLM](https://vllm.ai/) and [SGLang](https://docs.sglang.io/), both of which
flatten the frontend/backend divide and provide both the network-facing and
request routing and batching part, as well as the kernels to interface with the
hardware and execute the models. vLLM and SGLang do not have an explicit
compilation step, like TensorRT, but instead perform a degree of optimization
at startup to execute the model as a pipeline of kernels, which can be cached
and reused. Both of them are distributed with hand-coded kernels, such as
[FlashAttention](https://github.com/Dao-AILab/flash-attention) and
[FlashInfer](https://flashinfer.ai/), and can also leverage PyTorch’s
[torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html),
to fuse operations of the original model and replace them with optimized
kernels.

