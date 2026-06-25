---
title: "Appendix: GPU Hardware"
weight: 101
description: "Architecture, CUDA and ROCm, kernels and Triton, memory hierarchy"
---

Graphics Processing Units (GPUs) are massively parallel computing devices
capable of running many thousands of computations in parallel. Modern GPUs have
primarily two different types of cores: _CUDA Cores_[^cuda], which are
responsible for vector operations, and _Tensor Cores_[^tensor], which are
responsible for matrix multiplication.

Tensor cores are a relatively recent addition to GPUs and have many similarities
with Google's [Tensor Processing Units
(TPUs)](https://dl.acm.org/doi/10.1145/3079856.3080246). As a first
approximation from an LLM perspective, tensor cores can be thought of as
responsible for the matrix multiplication during attention scoring and the
feed-forward network, while the CUDA cores can be thought of as responsible for
normalization, softmax, and positional encodings.

GPU cores are much simpler than CPU cores and are stateless from an
architectural point of view. Instead, they are organized into _Streaming
Multiprocessors_ (SM)[^sm], which consist of the cores, registers and caches to
store state, as well as a scheduler. In many ways, an SM is more akin to a CPU
core, albeit with a huge amount of simultaneous multithreading (SMT), than the
GPU cores are.

The basic unit of execution on a GPU is, similarly to CPUs, called a _thread_.
GPUs, however, are Single Instruction, Multiple Thread (SIMT) devices, which
means that a single thread is not an independently schedulable entity like a CPU
thread. Instead, threads are grouped into _Warps_[^warp], which are the minimum
unit of execution and scheduling on the hardware.

A single Warp has [32
threads](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture),
running in lockstep and executing the same program on different pieces of data.
If there is a branch in the executing program, all the threads normally take
the same path; if threads want to take different paths, they are executed
sequentially in two groups, each taking one of the paths (with the other group
disabled and idling). This phenomenon, known as _branch divergence_,
significantly reduces utilization and hurts performance. Historically, one
could think of there being a single logical instruction pointer for the entire
Warp; modern architectures have per-thread instruction pointers, but the
execution of divergent threads is still serialized and harms performance.

Each Warp is assigned to a single SM, while all the Warps assigned to the SM
share the SM's cores, registers, and caches, with each thread getting a
statically partitioned subset of the register file. The Warps are scheduled onto
the SM's cores using a hardware scheduler called the _Warp scheduler_. This is
necessary because threads can block on memory accesses; consequently, SMs are
usually assigned more Warps than the actual number of cores available.

While a Warp is the hardware abstraction for a group of threads, the programming
model groups threads together into _thread blocks_ or a _Cooperative Thread
Array (CTA)_. Threads in the same thread block are guaranteed to execute on the
same SM at the same time; the runtime and hardware decompose the blocks into the
required number of Warps for execution. Threads within the same thread block can
coordinate access to memory, which is the primary mechanism for communication
between the threads, and synchronize using barriers.

Threads run programs called _kernels_, which are written to the specific
programming model offered by the hardware[^model]. More recently,
[Triton](https://triton-lang.org/main/index.html) allows kernels to be written
in a hardware-agnostic way in a Python-like DSL and compiles them to CUDA or
ROCm.

GPUs are capable of a much greater degree of parallelism than CPUs, but at the
cost of generality. The kernels largely express parallelizable transformations
on data, but usually do not encode complex control flow or decision-making
logic. Consequently, the GPU remains very much a subordinate device to the CPU,
which acts like the controller to determine what work is delegated to the GPU
and is able to manage memory as well as launch and terminate kernels on the GPU.

## Memory Hierarchy

The fastest storage available to GPUs is the _register file_, which serves the
same purpose as CPU registers in storing data that the cores can directly
access and manipulate without blocking. Unlike CPU registers which are attached
to cores, they are linked to the SM and assigned to threads; this allows
registers to be partitioned unevenly across threads executing different kernels
to better fit each kernel's requirements.

The next fastest storage is the L1 cache, which is around an order of magnitude
slower than the register file. The cache is built using low-latency _Static
Random Access Memory (SRAM)_ and is sometimes just referred to as the SRAM. The
cache is private to each SM and shared across the cores and serves the same
role as the L1 cache for CPUs. The primary difference from CPU caches is that
it is a hybrid of a software-managed scratchpad referred to as _Shared Memory_,
that requires data to explicitly be loaded, and a transparent hardware-managed
cache.

The next layer of the memory hierarchy is an entirely hardware-managed L2 cache.
It is shared across all the SMs and is a further order of magnitude slower and a
few orders of magnitude larger than the L1 cache.

The final layer is the main memory (_Dynamic Random Access Memory (DRAM)_) of
the GPU, historically referred to as _Video RAM (VRAM)_. The most common type of
VRAM in datacenter GPUs is called _High Bandwidth Memory (HBM)_. It is much
larger and much slower than the caches and its size and availability determines
the size of model that can be run and the degree of concurrency and context
sizes supported, since the model weights and inference cache (KV cache) reside
here.

This is a very high-level view of the GPU's hardware and programming model,
which is discussed in much greater depth in [Modal's GPU
Glossary](https://modal.com/gpu-glossary).

[^cuda]: Sticking to using NVIDIA's terminology here. The equivalent for AMD
    hardware are called _Stream Processors_.
[^tensor]: _Matrix Cores_ for AMD.
[^sm]: _Compute Units_ for AMD.
[^warp]: _Wavefront_ for AMD.
[^model]: CUDA on NVIDIA and ROCm on AMD.
