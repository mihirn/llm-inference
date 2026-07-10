---
title: "I/O-Aware Kernels"
weight: 5
description: "FlashAttention, FlashInfer, and Blockwise Parallel Transformers"
---

This section assumes an understanding of the GPU's memory hierarchy, a brief
refresher of which is at [Appendix: GPU Hardware](../hardware/).

## The Memory Bottleneck

GPUs consist of a relatively slow, large main memory (HBM) in which the model
weights and KV cache reside and a much faster, smaller set of caches (SRAM) on
which the cores actually operate. The primary bottleneck during attention is
the constant fetching and writing of data between the slow-but-large HBM and
the fast-but-small caches. Taking a naive attention implementation, with a
sequence of length $k$, the Q and K matrices, which are ($k \times d_k$), must
be loaded from HBM and $Q \times K^T$ materializes a ($k \times k$) matrix
which is written back to HBM. This is then fetched to calculate the attention
scores (using a softmax function) and written back to HBM; following which the
intermediate matrix and V, which is ($k \times d_k$), are loaded from HBM and
multiplied, resulting in a ($k \times d_k$) matrix written back to HBM. This
back and forth also forces the entire ($k \times k$) attention score matrix to
be materialized and written to HBM, which is a significant capacity and
bandwidth bottleneck for long sequences.

## FlashAttention

[FlashAttention](https://arxiv.org/pdf/2205.14135) speeds the attention process
by fusing the entire calculation into a single kernel. It uses a [tiled matrix
multiply](https://siboehm.com/articles/22/CUDA-MMM) to fetch small blocks of
the Q, K, and V matrices from HBM and operate on them using an [_online
softmax_](https://arxiv.org/pdf/1805.02867) calculation, so that the $Q \times
K^T$, softmax, and multiplication with V all occur in one pass, without ever
materializing the entire ($k \times k$) attention score matrix. The final
result is then written out to HBM.

Blocks of the Q matrix are fetched from HBM to the GPU’s caches and registers,
and then the K and V matrices are streamed through sequentially, in a blockwise
pattern. The partial blockwise results are aggregated in the cache before
finally being written out to the HBM. The ability to calculate the attention
scores in a streaming fashion is enabled by online softmax: typically,
attention scores require the entire row to be materialized so that it can be
summed and all the values scaled accordingly. Online softmax allows the softmax
calculation to proceed blockwise in a left-to-right manner in every row by
tracking the current sum and max value encountered and rescaling the partially
aggregated results accordingly. While the K and V matrices are fetched
repeatedly (once for every block of Q), the attention score matrix is never
written back and read, which significantly reduces the memory bandwidth
required.

FlashAttention was originally envisaged to be a training-side optimization, but
has increasingly become valuable during inference, particularly to speed up the
prefill stage, as prompt and context lengths increase. It has seen [iterative
advancements](https://github.com/Dao-AILab/flash-attention) which provide
better performance using better load balancing across the tensor cores or by
using specialized features of new hardware.
[FlashInfer](https://flashinfer.ai/) is a set of inference-optimized kernels
which implement FlashAttention while supporting multiple paged KV-management
systems, such as PagedAttention or RadixAttention, as well as support for
multi-head attention and grouped-query attention.

## Blockwise Parallel Transformers

[Blockwise Parallel
Transformers](https://dl.acm.org/doi/10.5555/3666122.3666508) (BPT) extend the
blockwise computation of FlashAttention to the feed-forward network of the
transformer. FlashAttention already divides the input sequence into blocks, and
for every Q block, it fetches all the K and V blocks and calculates the
attention output. The attention outputs for all the Q blocks, however, are
concatenated into sequence-sized activations for the feed-forward network and
fed into it in their entirety. This places significant demands on GPU memory,
as the requirements scale linearly with the input sequence length.

BPT is a scheduling optimization based on the observation that having a barrier
at the end of the attention phase is unnecessary and that the feed-forward
phase can also be executed blockwise. This is because the attention phase is
the only part of the transformer where cross-token information is needed; in
contrast, the outputs of the feed-forward network for every token (and hence
block) are independent of each other.

BPT still requires attention for a specific Q block to be completed using
FlashAttention; however, unlike before, it does not require the attention
output for _all_ the blocks before proceeding to the feed-forward network. This
allows the kernels to only materialize activations for the block, rather than
for the entire sequence, which reduces the per-layer activation memory
footprint by up to 4x, allowing larger sequences to be processed on the same
GPU.

## Additional References

1. [How FlashAttention Accelerates Generative AI Revolution (Video)](https://www.youtube.com/watch?v=gBMO1JZav44)
1. [Reimplementing FlashAttention for performance and giggles](https://aminediro.com/posts/flash_attn/)
1. [Flash Attention From Scratch](https://lubits.ch/flash/)
