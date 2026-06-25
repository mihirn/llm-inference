---
title: "Sharding a Model"
weight: 3
description: "Pipeline, tensor, and expert parallelism"
---

Model sizes and context windows have grown much faster than the memory capacity
of GPUs. As a consequence, serving an advanced model necessitates sharding the
model across multiple GPUs. Multi-GPU environments can be both intra-node and
inter-node. In intra-node setups, multiple GPUs in the same node are typically
connected with a dedicated hardware interconnect
([NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) for NVIDIA, Infinity
Fabric for AMD) supporting bandwidths of \~1TB/s.

Inter-node setups can use a dedicated interconnect for a few hundred GPUs
([NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/) for NVIDIA,
[UALink](https://ualinkconsortium.org/) for everyone else), but this is much
harder to scale and inter-node setups often end up relying on a low-latency
network interconnect like InfiniBand. The kind of interconnect available often
plays a role in determining the sharding strategy for the model.

The simplest sharding strategy is [_pipeline
parallelism_](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_simple.html)
in which different layers of the model are stored on different GPUs. A request
is processed entirely by a layer on one GPU and then forwarded to another GPU
for the next layer. Each GPU only needs to forward the contexts to a single
other GPU, resulting in point-to-point communication patterns. This approach
increases latency because a single request must traverse all the GPUs, only one
of which is actively processing the request at any given time. The lack of work
for the other GPUs in the pipeline, called pipeline bubbles, is addressed by
batching requests, which allows more simultaneous processing.

Another way of sharding the model is [_tensor
parallelism_](https://huggingface.co/blog/qgallouedec/tp) in which the model
weights of every layer are sharded across multiple GPUs. This may be different
heads in the attention layer, the feed-forward network, or both. Since each GPU
has only partial results which need to be combined, a specific type of
scatter-gather and reduction operation, called AllReduce, is required at the
end. Each GPU is responsible for a chunk of the final output, i.e., the
activations of the layer, and collects and reduces the partial sums for that
chunk from all other GPUs. The completed chunks are then simultaneously
broadcast across all the GPUs, so every device ends up with a complete output of
the layer, which acts as the input for the next layer.

Tensor parallelism has a much higher communication-to-compute ratio than
pipeline parallelism, making the latter preferred in inter-node deployments.
Within a single node with NVLink, however, tensor parallelism is often
preferred because it allows all the GPUs to be active simultaneously, resulting
in lower request latency. Further, if the model is large enough that even a
single layer is too large to fit on any of the GPUs, tensor parallelism is the
only available option.

Mixture-of-Experts (MoE) models can be deployed using [_expert
parallelism_](https://arxiv.org/pdf/2201.05596). During inference of an MoE
model only the top-$k$ experts are activated during the feed-forward network
phase, and a single expert is either fully activated or entirely idle. This
makes experts a convenient unit for partitioning where the weights for a single
expert are kept as close as possible (on a single GPU if they fit, else ideally
within a node), while other experts can be distributed across the cluster. Since
inference for each token activates different experts on different GPUs, the
tokens need to be sent to the appropriate GPUs and the outputs returned before
layer normalization can proceed. This results in cluster-wide communication
patterns, where load imbalances which slow any expert stall the entire inference
pass.

Pipeline, tensor, and expert parallelism are distinct strategies, but can be
used together in the same deployment; for example, a large model may use
pipeline or tensor parallelism for its attention phase and expert parallelism
for the feed-forward layer, while a model using expert parallelism with large
experts may use tensor parallelism to split a single expert across multiple
devices.

