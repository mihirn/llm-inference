---
title: "Appendix: Recent Research"
weight: 103
description: "Notes from recent papers"
---

These are some notes from recent papers which don't fit neatly into any of the
sections because they require more context and framing to bridge the gap from
the earlier, more fundamental works. The notes are less polished than the
chapters and the hope is to eventually weave them into the broader narrative,
but until then they'll be collected here.

## Routing and Scheduling

[_LMetric_](https://www.usenix.org/conference/osdi26/presentation/zhang-dingyan)
(_OSDI'26_) is a request routing/load-balancing system from Alibaba which has
been deployed and serves production traffic. The deployment model is assumed to
be with prefill-decode colocation and chunked prefills. Every instance has the
entire model and can serve any request. In practice, this may be a group of
hosts or GPUs the model is sharded across; however, from the perspective of
traffic distribution it can be thought of as a single instance.

LLM routing is a classic data-aware routing problem which has to balance the
twin concerns of locality and load. Serving instances keep the KV cache around
after servicing the request for some time so that any potential follow-up
requests get a prefix cache hit, which saves a significant amount of compute
(cached prefixes are around a tenth of the cost of new input tokens at frontier
labs' API pricing). This is attractive for both multi-turn conversation
chatbots and coding agents; on the other hand, purely prefix-cache based
routing ends up with hotspots overwhelming individual instances.

There is a long body of work trying to balance these two parameters using
threshold functions to move between data and load-based routing (called
filter-based combinations in the paper) or using simulation to model the cost
of recomputing prefill tokens and using that for decision making. Prior work
from Alibaba extends basic shortest-queue routing with a scoring function for
prefix hits, which uses hyperparameters to determine how heavily to weight
prefix matches vs load.

All these methods require knowledge of the underlying model and hardware and
need to be tuned extensively to work well. In contrast, LMetric uses a simple
hardware- and model-agnostic function which is just a multiplication of the
_estimated number of prefill tokens_ (P) that an instance would have to
calculate, including already queued requests and accounting for prefix cache
hits, and the number of active requests at the instance, called the _batch
size_ (BS). The instance with the lowest ($P \times BS$) score is chosen since a
low score indicates either less recomputation, lower load, or both.

This is almost deceptively simple with only two variables and no hyperparameter
tuning, but works surprisingly well for the evaluated workloads. The
pathological case is where some instances are repeatedly scored low, for
instance, with near-complete prefix cache hits causing P and the score to be
almost zero regardless of load, while the recomputation cost is high on other
instances, so that all requests end up getting sent to the same few instances
and overload them. The system detects and temporarily excludes hotspots or
shifts to queue-length based routing in these cases.

## GPU Sharing

[_Prism_](https://www.usenix.org/conference/osdi26/presentation/yu-shan)
(_OSDI'26_) is a GPU memory management system designed to increase GPU
utilization in multi-model serving workloads, such as those seen at
Token-as-a-Service providers. These providers serve several models in parallel
across their GPU fleet with models sharing GPUs; production traces reveal that
the models tend to receive requests in short bursts punctuated with long idle
intervals and that the popularity of models fluctuates over time. Time sharing
approaches focus on efficiently swapping in and out model weights and violate
SLOs when multiple models have overlapping bursts, while static space sharing
focuses on efficiently sharing capacity for the KV cache across models and
suffers from SLO violations during bursts since idle capacity cannot be reused.

Prism combines both time and space sharing approaches. It allows memory to be
shared across different models for either model weights or the KV cache by
virtualizing the physical address space of the GPU into a number of contiguous
address spaces for the inference engines. It operates at page granularity and is
agnostic to the contents of the virtual memory page, which is controlled by the
engine; however, pages are not shared across models, which can result in
internal fragmentation. Weights of inactive models are swapped out from GPU
memory to host CPU memory.

Swapping models back in on demand can violate SLOs, so Prism optimizes this path
by pre-allocating engines as well as reusable virtual address spaces for
different models, rather than creating a new engine instance from scratch.
Further, if multiple GPUs are available, it parallelizes loading model weights
into GPU memory by partitioning the load across multiple GPUs in the same node
to maximize the host-to-GPU bandwidth and then collecting the weights at the
target GPU using NVLink interconnects. The KV capacity of other models is
reduced to accommodate the swapped-in model, similar to classical memory
ballooning techniques.

Finally, Prism is responsible for placement and scheduling decisions. It
periodically sorts the models in descending order of their resource demands,
calculated as a combination of load, KV size, and the TPOT SLO, and greedily
assigns models to GPUs to reduce KV pressure. For scheduling, it holds a single,
shared per-device request queue, and prioritizes running the maximum number of
requests that can meet their TTFT SLO.
