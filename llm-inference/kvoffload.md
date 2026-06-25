---
title: "KV Cache Management and Offload"
weight: 8
description: "Prefix caching and KV offload"
---

The only required long-lived state during LLM inference is the model, as every
token in the response to a request can be regenerated from scratch if required.
The KV cache is one method to trade compute for memory (and additional state),
by storing the K and V matrices for every token in the response; however, the
KV cache is assumed to be discarded once the response is complete.

## Prefix Caching

This one-and-done model does not represent real workloads: requests are not
independent, but often are follow-ups from previous requests. Examples of this
include chat and coding sessions or even business intelligence over documents
and reports, where a large part of the context is reused across requests. Even
in cases where the requests are from different users, the context may be
reused; for instance, where a large preamble about expected model behaviour is
attached to every request. _Prefix caching_ (or _prompt caching_) allows parts
of the KV cache to outlive a request and be reused across multiple
non-concurrent requests and significantly reduces response latency.

[RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/), introduced in
SGLang, is a radix tree/trie-based indexing structure to help index the cached
prefixes. Requests are added token-by-token to the tree with each node pointing
to the associated KV cache entry; when a new request arrives, it walks the tree
to find the maximum length of a matched prefix and then starts the prefill
operation. Since the radix tree does a token-by-token exact match, even minimal
reordering or changes to the prompt cause a cache miss.

As contexts get larger, caching the prefix in HBM reduces concurrency, since
the more memory that is used for caching, the less can be used for request
decoding. Further, as the contexts for requests get larger, the requests
themselves require more memory for token generation, creating even more memory
pressure. While context truncation techniques like attention sinks help shrink
the KV cache for active requests, the cached prefixes also need to be handled,
potentially by offloading to external memory, with RadixAttention using a
simple LRU eviction policy.

## Cache Offload

[LMCache](https://arxiv.org/abs/2510.09665) is a system that allows inference
systems, such as vLLM and SGLang, to delegate cache offload responsibilities to
an external system. It allows the KV cache to be stored in host memory, on
stable storage, or on remote hosts to free up GPU memory. LMCache has a couple
of features: firstly, the KV cache for a request is scattered across memory due
to paging, so directly copying it out involves a number of small, slow I/O
operations. Instead, LMCache aggregates data in a buffer in the HBM which is
then written out in a larger batched operation to improve performance.
Secondly, LMCache fetches the KV cache layer by layer, while overlapping the
I/O with compute, so that the request does not stall during execution.

LMCache also defines a fairly general interface to fetch and store the KV and
to query whether the prefix is a hit in cache. This allows most inference
engines to add support for LMCache without requiring LMCache itself to be aware
of the different KV cache setup for different engines and model architectures.
LMCache is primarily a vLLM extension though and handily outperforms both
vanilla vLLM and vLLM with its own internal CPU offloading; however, SGLang’s
own CPU offloading matches its performance, which suggests vLLM’s performance
problems are implementation issues. vLLM recently added support for a
[cross-layer KV layout](https://github.com/vllm-project/vllm/pull/27743), which
allows offload to operate on large, contiguous blocks; vLLM should be
re-evaluated with this enabled before making any performance conclusions
vis-à-vis LMCache or SGLang.

LMCache does not treat prefix matching as an all-or-nothing phenomenon and
instead divides the prompt into a number of chunks and searches for matches
using hashes. Context truncation techniques, which reduce the memory footprint
of the KV cache, significantly reduce the hit rate, since even identical
content prompts, with minor positional variations, will have discarded a
different set of tokens. Chunk-based hashing increases the hit rate, but
overall, cache offloading works best without truncation.

[CacheBlend](https://arxiv.org/pdf/2405.16444) extends the ideas of LMCache so
that the external cache can be used not just for an exact prefix match, but
also where subsets of the prefix match. This is common when data from multiple
sources is prepended to the query as part of the prompt; for example, multiple
queries over a number of documents may add the documents in different order or
have a slightly different number of documents as part of the context. Existing
systems chunk the prompt and either strictly match chunks in-order, until the
first deviation is detected, or match all possible chunks in any order and
reuse the KV cache for those tokens unchanged. The former approach preserves
quality at the cost of performance (lots of potentially matched chunks go
wasted), while the latter is faster, but at the cost of quality (since any
cross-token attention across chunks can get missed).

CacheBlend picks an intermediate point in the design space: it reuses as many
chunks as it can match, but recalculates KV values for a small number of tokens
with high cross-chunk attention (empirically determined to be \~15%), so that
quality is preserved. It identifies these tokens by recalculating KV values for
all the tokens for the first layer and comparing against the cached KV values;
tokens with the highest deviation are selected and their KV values calculated
for the next layer. This continues for every layer until the number of tokens
are winnowed down. The main intuition behind this is that cross-chunk attention
affects only a few tokens due to attention sparsity and there is very high
correlation between the affected tokens across layers, so those tokens can be
identified early. This allows CacheBlend to preserve quality while reusing most
of the matched prefixes with additional compute proportional to the number of
selected tokens.

