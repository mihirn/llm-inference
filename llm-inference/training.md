---
title: "Appendix: Overview of Training"
weight: 100
description: "Fine-tuning, RLHF, RLAIF, quantization, and alignment techniques"
---

## Pre- and Post-Training

Transformers form the basis for LLMs, but there are a number of steps that
occur in addition to training the weights of the transformer’s neural networks
for them to be useful. The process of learning these weights from a large
corpus of data is called _pre-training_; however, the resulting base model is
only capable of predicting the next token given a sequence of tokens, but is
not optimized to provide outputs useful to the user.

The steps that follow pre-training are called _post-training_. One such step is
_fine-tuning_, which is a common method to align the base model, by updating
its weights, towards a specific goal, such as to convert the next token
prediction mechanism into a chatbot which understands the kinds of responses
humans prefer (Instruct Tuning) or to improve safety.

_Supervised Fine-Tuning (SFT)_ trains the model using a human-curated
selection of examples of prompts and what constitutes a good response. For more
nuanced problems, a number of human scorers grade outputs which are used to
train a scoring (or reward) model to reflect human preferences. The base model
(or one that has already undergone SFT) is then further trained in a
reinforcement learning loop guided by the scoring model. This is called
_Reinforcement Learning from Human Feedback
([RLHF](https://huggingface.co/blog/rlhf))_; if the scoring model does not
incorporate human feedback directly it is called _Reinforcement Learning from
AI Feedback (RLAIF)_.

Fine-tuning all the parameters of the model is extremely slow and expensive.
Instead, techniques such as Low-Rank Adaptation
([LoRA](https://openreview.net/forum?id=nZeVKeeFYf9)) freeze the original model
and augment it with small matrices (the LoRA matrices), which often have \< 1%
of the parameters, and are trained. The size of the LoRA matrices is called
their rank; if the fine-tuned model is not satisfactory, the rank can be
increased and the fine-tuning repeated. Once complete, the LoRA matrices can
be loaded alongside the original model during deployment or they can be
merged with the model so the final deployed model does not have additional
latency.

## Quantization

Quantization reduces the overall memory footprint of the model by
trading model accuracy for size by reducing the precision of the weights.
Common quantizations include 4-bit (INT4) and 8-bit (INT8) integers, though not
all hardware supports all quantization-levels. Two common quantization methods
are [_GPTQ_](https://arxiv.org/abs/2210.17323) and
[_AWQ_](https://hanlab.mit.edu/projects/awq).

GPTQ quantizes the weights of each layer sequentially with a target of reducing
the difference between the quantized and unquantized models for each layer,
while AWQ focuses on identifying the most significant model weights and storing
them with higher precision, while reducing precision on the less important
weights. Both are data-dependent post-training processes and require sample data
to calibrate.

Not all quantization methods require calibration data:
[_HQQ_](https://dropbox.tech/machine-learning/halfquadratic-quantization-of-large-machine-learning-models)
and [_k-quants_](https://github.com/iuliaturc/gguf-docs/blob/main/k-quants.md)
both focus on reducing the errors in the values of the weights, rather than
their impact on activations, which can be achieved without any calibration
data. HQQ achieves this using an iterative solver at a fixed precision, while
k-quants use a mixed precision representation where weights are divided into
blocks, stored at lower precision, along with a base and a scaling factor
stored at higher precision.

While quantization is primarily used before deploying models for inference, it
can also be used while fine-tuning large models. Quantized LoRA (QLoRA)
fine-tunes a quantized version of the frozen original model, while keeping the
LoRA matrices unquantized, with the goal of producing LoRA matrices that are
comparable to those from fine-tuning the unquantized model.

