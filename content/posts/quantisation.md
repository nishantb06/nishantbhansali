---
title: "Quantisation Notes"
date: 2024-05-07T19:38:20+05:30
draft: true
---

[Lie mao blog very detailed with code ](https://leimao.github.io/article/Neural-Networks-Quantization/)
https://iq.opengenus.org/basics-of-quantization-in-ml/
[See for code for activation aware quantization](https://www.youtube.com/watch?v=XM8pllpBVA0)
[Hugging face quantisation](https://huggingface.co/docs/optimum/concept_guides/quantization)
[Pytorch quantisation api docs] 

There are 4 methods to optimise a model, GPTQ, activation aware quantized training, bits and bytes , packages like huggingfuace optimum, or pytorch api itself. 



Scripting the model can make inference faster for a few reasons:

1. **Reduced Overhead**: Scripted models can have lower overhead compared to their original Python counterparts because the script represents a more optimized version of the model's forward pass. This can lead to faster execution times.

2. **Optimizations**: When you use `torch.jit.script` to script a model, PyTorch applies various optimizations to the script, such as constant folding and operator fusion, which can improve performance during inference.

3. **Parallelization**: Scripted models can take advantage of parallelization opportunities more effectively, especially when deployed on hardware accelerators like GPUs, due to the way the operations are organized and optimized in the script.

4. **Serialization**: Scripted models can be serialized and deserialized more efficiently, which is important for deployment scenarios where models need to be loaded quickly into memory.

5. **Platform Independence**: Scripted models are platform-independent once they are compiled, meaning they can be executed on any platform that supports PyTorch without needing the original Python code, which can be beneficial for deployment in different environments.

Overall, scripting a model can lead to faster inference times due to these optimizations and efficiencies, especially in production environments where speed and resource usage are critical.

Tracing a model using `torch.jit.trace` and `torch.jit.script` and quantization are two distinct processes in PyTorch, each serving different purposes:

1. **Tracing**:
   - **`torch.jit.trace`**: This function takes an input tensor and traces the operations that occur during the forward pass of the model. It records the operations as a `ScriptModule`, which can be used for inference. Tracing is useful for models that have fixed control flow (i.e., the execution of the model does not depend on dynamic conditions like loops or if statements).
   - **`torch.jit.script`**: This decorator converts a Python function into a `ScriptModule`, allowing for more flexibility in defining the model's forward method. It is used when the model has dynamic control flow.

2. **Quantization**:
   - **Quantization** is the process of converting a model to use fewer bits to represent weights and activations, usually from 32-bit floating point to 8-bit integers (or even lower bit representations). This can significantly reduce the model size and improve inference speed, especially on hardware that supports low-precision operations efficiently.
   - PyTorch provides tools like `torch.quantization` module to help with quantization, including functions to prepare the model for quantization (`torch.quantization.prepare`) and to actually quantize the model (`torch.quantization.convert`).

In summary, tracing is about capturing the operations of a model to create a script representation for efficient inference, while quantization is about reducing the precision of the model's parameters and activations to improve performance and reduce memory footprint.