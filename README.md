# Quantization Examples in PyTorch

This repository contains examples of model quantization in PyTorch.

# Model Comparison

The table below summarizes a benchmark of different types of models.

| Model    | Dtype   |   Model Accuracy |   Inference Time |   Parameter Size |
|----------|---------|------------------|------------------|------------------|
| origin   | fp32    |           81.36% |          ~8.15ms |          ~42.8Mb |
| ptq-d    | int8    |           81.36% |          ~8.13ms |          ~42.8Mb |
| ptq-s    | int8    |           78.14% |          ~2.42ms |          ~10.8Mb |
| qat      | int8    |           78.85% |          ~2.45ms |          ~10.8Mb |

Note: all results are measured with the torch JIT model on a single CPU core.

Environment: Win10 22H2, Intel Core i5-10210U @ 1.60GHz 2.11GHz, 16GB RAM.

# Recommended Reading

1. For a brief introduction to model quantization, and the recommendations on quantization configs, check out this PyTorch blog post: [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/).

2. For detailed information on model quantization, including best practices, check out the PyTorch documentation: [Quantization](https://pytorch.org/docs/stable/quantization.html).

# Tutorials

- [Static Quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

- [Quantized Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html)

- [Dynamic Quantization on BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)

# References

- [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#device-and-operator-support)

- [PyTorch Dynamic Quantization](https://leimao.github.io/blog/PyTorch-Dynamic-Quantization/)

- [PyTorch Static Quantization](https://leimao.github.io/blog/PyTorch-Static-Quantization/)

- [PyTorch Quantization Aware Training](https://leimao.github.io/blog/PyTorch-Static-Quantization/)

- [Quantization Tutorials by Oscar Savolainen](https://github.com/OscarSavolainen/Quantization-Tutorials)
