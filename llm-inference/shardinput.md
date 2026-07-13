---
title: "Sharding the Input"
weight: 9
description: "Sequence and context parallelism, Ring Attention, and DeepSpeed-Ulysses"
---

As the input context (also referred to as the _sequence_) gets longer, the
requirement that the entire sequence fit in the memory of a single GPU starts
becoming a limiting factor. While model sharding techniques, such as tensor
and pipeline parallelism (discussed in [Sharding a Model](../modelsharding/)),
split the layers and heads of the model across multiple devices, each of them
still processes the entire sequence for the subset of the model on the device.
This both limits the maximum sequence size and increases response latency when
processing large sequences.

_Sequence parallelism_ (or _context parallelism_[^cp]) shards the sequence into
chunks and distributes them across multiple GPUs, so that every device is only
responsible for a subset of the entire sequence. While originally designed to
allow for training on longer sequences, it is increasingly used during
inference as well, particularly for prefill, as context windows for models
extend to hundreds of thousands or millions of tokens.

Despite the sharding of the sequence, attention still requires tokens from the
entire sequence to influence each other, so these devices communicate with each
other to exchange parts of the sequence. Broadly speaking, the communication
patterns for all the sequence parallelism techniques are either point-to-point
with devices arranged in a virtual ring, or all-to-all across the entire group
of devices.

In ring-based schemes, the actual computation structure is similar to
FlashAttention and Blockwise Parallel Transformers (both discussed in
[I/O-Aware Kernels](../flashattn/)), where the sequence is divided into blocks
and each Q block iterates over every K and V block for attention (like
FlashAttention) and is then followed by the feed-forward network (like BPT).
This allows every Q block to be processed independently of the other Q blocks.

[Ring Self-Attention](https://arxiv.org/abs/2105.13120) and its successor [Ring
Attention](https://proceedings.iclr.cc/paper_files/paper/2024/file/1119587863e78451f080da2a768c4935-Paper-Conference.pdf)
apply the same computation structure in a distributed setting: each device
holds only a subset of the entire sequence and is responsible for the Q, K, and
V blocks for that sub-sequence. Each device repeatedly transmits K and V blocks
to the next device in the ring to parallelize attention. After several
iterations, every device has seen all the K and V blocks and attention is
complete, which is followed by the feed-forward network. If the
per-block compute time exceeds transfer time, communication can overlap with
compute in a way that the subsequent iteration is usually not blocked waiting
for neighbouring blocks. The paper examines the compute and interconnect for
several GPU setups and concludes that this is true for even medium-sized blocks
on low-latency interconnects.

[Striped Attention](https://arxiv.org/abs/2311.09431) is a load-balancing
optimization for ring attention that distributes non-contiguous blocks, i.e.,
stripes of the input sequence to the devices. Causal masking in attention
prevents later tokens from influencing the attention score of earlier tokens,
which means that the work is not equally balanced across the sequence with some
block pairs entirely masked out for every round. Distributing contiguous
blocks, as traditional ring attention does, results in some devices having
block pairs masked out for most rounds (the devices with early blocks), while
others are unmasked for most rounds (the ones with late blocks). This unequal
distribution of work leads to stragglers and can waste up to half the available
compute. [DistFlashAttn](https://arxiv.org/abs/2310.03294) addresses this same
imbalance using a work-stealing style scheduler which redirects attention
computation from busy to idle devices.

Sequence parallelism is orthogonal to tensor and pipeline parallelism, and it
is entirely possible to combine them to shard both the model parameters and the
input sequence at the same time. A common pattern is to shard the attention
heads across multiple devices, using tensor parallelism, and then shard the
input and transfer its blocks amongst the tensor parallel groups using ring
attention.

[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) assigns each device a
subset of the sequence but performs attention by sharding the heads across the
devices. It uses an all-to-all communication pattern so that all the Q, K, and
V blocks of the sequence are sent to the device, but only for the heads that it
is responsible for; these heads then perform attention for the entire sequence
before another all-to-all communication redistributes the resulting activations
for the appropriate subsets of the sequence to their corresponding devices.

All-to-all communication reduces the amount of data being sent across the
interconnect compared to ring attention: each device is only sent the Q, K, and
V blocks for the heads it is responsible for, while in ring attention every
device gets the K and V blocks for every head. The downside, however, is that
model architectures with a small number of KV heads, as with GQA, limit the
degree of available parallelization and substantially reduce the size of the K
and V blocks. DeepSpeed-Ulysses also implicitly assumes that Q, K, and V for
the entire sequence, for at least a single head, can fit on a single device.

[Unified Sequence Parallelism](https://arxiv.org/abs/2405.07719) (USP) combines
both DeepSpeed-Ulysses and ring attention into a single, flexible system. The
devices are divided into a number of groups, where each group runs all-to-all
communication internally, while cross-group communication follows ring
attention. If there is only one group, it is effectively DeepSpeed-Ulysses,
while if each group has only one device, it is effectively ring attention. This
allows USP to work around model head and interconnect limitations: for example,
scaling beyond the number of KV heads of the model or using all-to-all
communications over high-bandwidth interconnects, such as NVLink, and ring
communications over lower bandwidth interconnects, such as InfiniBand.

Sequence parallelism takes a different form during the decode phase of
inference, where the system is processing a single Q token, alongside the
entire KV cache on every iteration. When prefill is sharded across devices, the
KV cache will naturally end up distributed; however, transmitting the entire KV
cache repeatedly for every new token is massively inefficient. Instead, [Ring
Pass-Q](https://arxiv.org/abs/2411.01783) flips the communication pattern and
distributes Q, in a ring pattern, to all the devices which process it locally
to compute partial attention. The partial attentions need to be merged for the
final attention output, which requires an all-to-all communication, but only
for the attention output of a single token—an additional step compared to
ring attention, where no final coordinated reduce step is required. The system
can dynamically move between distributing the KV cache and the Q embeddings
depending on the number of heads, sequence size, and available bandwidth, but
in practice, distributing the Q embeddings during decode is almost always the
right choice for long inputs on large models.

[^cp]: NVIDIA uses context parallelism to refer to the kind of sharding
    described and reserves sequence parallelism to refer to a much more limited
    form of sharding the input, where only certain [non-attention layers are
    distributed](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html).
