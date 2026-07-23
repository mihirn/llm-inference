---
title: "Batching, Scheduling, and Paging"
weight: 4
description: "Continuous batching, Orca, and PagedAttention"
---

Serializing requests and running them one at a time is incredibly wasteful.
Modern GPUs are extremely parallel devices and require several requests to be
run concurrently to fully utilize the underlying hardware. As soon as the
inference runtime has to handle multiple concurrent requests it faces similar
challenges to traditional systems in terms of resource partitioning: _which
request should be run, when, and for how long? How much memory should a single
request be allowed to consume?_ Classical systems concepts such as batching,
scheduling, and paging have been adapted to inference runtimes and are just as
valuable as they are in classical systems.

## Batching and Scheduling

Executing the model’s kernels on the GPU can generate one or more new tokens
before returning control to the CPU, either via an interrupt or through
polling. For a batch of requests, which often have significant differences in
output length, running to completion has a couple of issues: requests in the
batch are gated by the longest/slowest to complete request and newly arrived
requests suffer from head-of-line blocking while the batch is executing.

[Orca](https://www.usenix.org/system/files/osdi22-yu.pdf) introduced the idea
of iteration-level batching and scheduling, where the CPU creates a batch of
requests and then runs the model’s kernels for only a single iteration, i.e.,
to generate only a single token for all the requests. This allows the batch to
be adjusted after every iteration with completed requests removed and newly
arrived ones added. Switching execution between the CPU and GPU and
reconstructing the batch on a per-token basis is called _continuous batching_
and has become the predominant execution model for all the major inference
runtimes.

Switching execution between the CPU and GPU increases scheduling overhead,
which has typically been ignored because the cost of token generation has
dwarfed the scheduling time. However, optimizations have sped inference up
enough that this scheduling overhead can be substantial ([up to
50%](https://mlsys.wuklab.io/posts/scheduling_overhead/)) for smaller models.

There are a couple of other challenges with iteration-level batching: firstly,
if in-flight requests, each of which requires the HBM for their KV cache, are
constantly added and removed from the batch, memory could be allocated in a way
that no request can complete and the system deadlocks. Orca solves this by
pre-allocating memory sized to the maximum allowed response for every request
and only allowing new requests to be added to the batch if there is available
HBM for the entire response. Further, it also prioritizes request execution in
the arrival order, which also helps overall latency.

The other challenge with batching is that the attention kernels on the GPU can
only operate on a batch if all the requests have the same shape and desired
operations, i.e., they should all be either in the prefill stage or the
autoregressive stage and should have the same number of tokens (input tokens in
the prefill case, cumulative input and generated tokens in the autoregressive
case). This would allow the Q, K, and V vectors to have the same dimensions and
parallelize matrix multiplication. This is not feasible for any real workload,
so Orca parallelizes operations on the GPU which don’t require everything to be
the same shape (normalization, addition, ReLU/GeLU) and then splits the matrix
prior to attention, which proceeds serially. This is evaluated in the paper and
affects performance less than one might expect; a consequence of the fact that
attention is memory-bound, rather than compute-bound, and the KV cache for each
request is unique, meaning batching multiple requests does not significantly
improve the throughput of the attention phase.

## GPU Paging

Orca pre-allocates memory for the maximum allowed response length for every
request. This is wasteful since the vast majority of requests will terminate
before reaching the maximum length.
[vLLM](https://dl.acm.org/doi/10.1145/3600006.3613165) introduced
_PagedAttention_, an idea which borrows from pages in traditional virtual
memory, to reduce the inefficiency caused by internal fragmentation. In
PagedAttention, the KV cache is no longer stored in contiguous physical memory
on the GPU; instead, memory is divided into a number of fixed-size blocks (or
pages), and a page table which maintains the mapping between the blocks and
physical memory addresses.

Each request starts by allocating the number of blocks which can accommodate
the prompt, after which new blocks are only allocated when the previous block
is full. The newly allocated block is given the next logical address but is not
required to be contiguous in physical HBM. Existing GPU kernels expect to run
on physically contiguous memory, so vLLM implements specialized kernels which
are aware of the page tables and perform block reads and writes through the
layer of indirection provided by them.

The page table which handles the mapping between logical and physical blocks is
managed by the CPU, which is responsible for allocating new blocks when blocks
are full or freeing blocks associated with completed requests. The CPU is also
responsible for copying the page tables for all requests in a batch into HBM
before starting to execute the batch on the GPU.

Since vLLM does not pre-allocate the HBM, it runs the risk of deadlock when all
the memory is exhausted and no requests can proceed. In this case, it opts to
evict a request from the HBM (either the lowest priority request, if [priority
scheduling](https://github.com/vllm-project/vllm/pull/5958) is enabled, else
the most recently arrived one, since it defaults to a FCFS policy), either to
CPU memory (like swapping to disk) or by discarding the data and forcing the
request to be recomputed from scratch.

Decoupling logical and physical memory has benefits beyond just reducing
internal fragmentation: it allows blocks to be shared across multiple requests.
This is useful in cases where a single prompt might generate multiple responses
or when requests have long shared prefixes. Shared blocks should outlive single
requests and only be evicted when all the requests relying on them have
finished; vLLM handles this by reference counting the blocks in the page table.

[Pensieve](https://dl.acm.org/doi/abs/10.1145/3689031.3696086) generalizes
PagedAttention to support non-contiguous memory regions during both the prefill
and decode phases. The original kernel only supports single-token attention
during the decode phase, while prefill occurs using a separate kernel which
relies on the KV cache being contiguous in GPU memory. By supporting paged
memory for both prefill and decode, Pensieve also relaxes the scheduling
requirement that a single batch contain only prefill or decode requests,
instead allowing requests in different stages to be merged in a single batch to
improve utilization and reduce head-of-line blocking. Mixed prefill-decode
batches are now broadly supported in most of the major inference runtimes.
