---
title: "Prefill-Decode Scheduling and Disaggregation"
weight: 7
description: "Chunk prefill and prefill-decode disaggregation"
---

The contrast between the resource and performance characteristics of the
prefill and decode stages presents both challenges and opportunities: prefill
can saturate the GPU with even a single long prompt and processes several
tokens in parallel, while decode is memory-bound, incapable of saturating the
GPU, and emits tokens serially. Speculative decoding attempts to increase
efficiency and reduce latency by making the decode phase look more like
prefill, since verification is essentially a prefill-like parallel phase.

For a single request, prefill has to be completed before decode can commence;
however, the ordering of prefill and decode across multiple requests can be
tuned to better prioritize throughput and resource efficiency or response
latency. Schedulers can choose to prioritize batches of entirely prefill or
decode operations or run mixed prefill-decode batches. Prioritizing prefill or
mixing prefill-decode batches increases resource efficiency at the cost of
latency since decode stages are either slowed down or stalled by the prefills.
Conversely, schedulers which prioritize decode-only batches before allowing any
new prefills to occur have better latency but worse throughput.

## Chunked Prefills

[Sarathi](https://www.usenix.org/system/files/osdi24-agrawal.pdf) attempts to
balance this trade-off by not taking an all-or-nothing approach to prefill and
instead dividing the initial prompt into a number of chunks and running mixed
prefill-decode batches. This approach is called _chunked prefills_, and the
size of the chunk is determined by the maximum number of tokens the GPU can
process in a batch while achieving its SLO, as well as the number of decode
requests already in the batch. The downside is that prefill is a naturally
parallelizable operation that could be completed in a single pass. Dividing it
into smaller chunks is less efficient both because it operates on smaller
matrices which make saturating the GPU harder and because fetching model
weights and populating the KV cache multiple times increases memory bandwidth
pressure. Consequently, if the chunk sizes get too small, the operation becomes
memory-bound and throughput and efficiency suffer.

## Prefill-Decode Disaggregation

[DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) takes
a more extreme approach to preventing prefill and decode from interfering with
each other by moving them to separate devices: most systems co-locate prefill
and decode on the same device because it allows both model parameters and the
KV cache to be reused which preserves precious HBM. DistServe turns this on its
head by replicating the model parameters and instead running prefills on one
GPU and then shipping the populated KV cache for a request over to another GPU
which runs the autoregressive phase. This is not actually a huge deal because
moving the KV cache either within a node or across nodes using a low-latency
interconnect is significantly faster than prefill and decode, resulting in only
a small impact on request latency (0.1% with a p95 \< 30ms).

DistServe is targeted towards environments with multiple GPUs, both within a
single node, and across multiple nodes, where latency is the key consideration.
Disaggregating prefill and decode also simplifies scheduling (no longer any
need to balance prefill or decode prioritization), and allows resources to be
scaled independently for either task. The core of DistServe is a placement
engine which determines the number of prefill and decode instances and how they
should be situated on physical hardware. While DistServe is able to get rid of
all prefill-decode interference, it does little to improve the utilization of
the decode resources.

Prefill-Decode disaggregation is available in
[most](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
[major](https://github.com/sgl-project/sglang/issues/4655)
[serving](https://docs.nvidia.com/dynamo/latest/design_docs/disagg_serving.html)
[systems](https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/prefill-decode.html).
[Splitwise](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/)
explores disaggregation from the perspective of a hosting provider with
heterogeneous fleets and how disaggregation allows prefill and decode to run on
the hardware best suited to their specific resource needs—for instance, decode
requires lots of memory and little compute and can be run on GPUs with fewer
tensor cores or potentially even CPUs with large memory capacity.

Efficiently transferring the KV cache between the prefill and decode machines
is key to getting reasonable performance in disaggregated systems. DistServe
takes a pull approach where the KV cache lives in the memory of the prefill
nodes and is pulled by the decode nodes just before they are ready to start
processing the request. Splitwise overlaps prefill computation with KV transfer
by eagerly transferring the KV cache for different layers as they get
calculated, while [DejaVu](https://proceedings.mlr.press/v235/strati24a.html)
also proposes prefill-decode disaggregation as a means to address the bimodal
nature of prompt processing and token generation. It focuses on the design and
implementation of a layer-by-layer streaming KV management library which swaps
the KV cache out of HBM to system memory or across a low-latency network to
other devices between the prefill and decode stages. DejaVu also observes that
the same mechanisms for KV cache offload and transfer allow for replication and
for the system to provide fault tolerance as needed.

Prefill-decode disaggregation is not the only attractive disaggregation target:
[the attention and the feed-forward
network](https://hao-ai-lab.github.io/blogs/distserve-retro/) also have vastly
different resource and performance characteristics and are potential future
areas of research.

