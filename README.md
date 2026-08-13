# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# thinh phan's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/thinhphan130997.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Convolutional Block | Implement a ResNet convolutional block with a projected shortcut that matches changed spatial and channel dimensions. | https://www.tensortonic.com/research/resnet/resnet-conv-block |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Cross Entropy Loss (Mean Reduction) | Implement mean categorical cross-entropy in Triton with stable row-wise log-sum-exp and atomic loss accumulation. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-cross-entropy |
| Dropout (Inverted Scaling) | Implement inverted dropout in Triton with a supplied mask, register scaling, and tail-safe tiled memory access. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-dropout |
| Fused Matmul + Bias + ReLU | Fuse tiled matrix multiplication, per-column bias, and ReLU in one Triton kernel with tail-safe memory access. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-fused-matmul-bias-relu |
| Fused Multiply-Add | Implement a Triton fused multiply-add kernel with contiguous tiles, hardware FMA, and masked tail handling. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-fused-multiply-add |
| Fused Row-Wise Softmax | Implement fused row-wise softmax in Triton with stable max subtraction, register reductions, and masked column tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-fused-softmax |
| GELU | Implement exact GELU activation in Triton with device error-function math and masked contiguous tiles. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-gelu |
| GEMV: Matrix Vector Product | Implement Triton matrix-vector multiplication with row-block programs, float32 accumulation, and masked matrix tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-gemv |
| L2 Vector Norm | Compute a Triton L2 vector norm with tiled sum-of-squares reduction, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-l2-norm |
| LayerNorm Forward | Implement LayerNorm forward in Triton with per-row mean and variance reductions, affine parameters, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-layernorm |
| LayerNorm Backward | Implement LayerNorm backward in Triton with row-wise statistics and atomic gradients for scale and bias. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-layernorm-backward |
| Row-Wise LogSumExp | Implement numerically stable row-wise LogSumExp in Triton with max subtraction and masked register reductions. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-logsumexp |
| Tiled Matrix Multiplication | Implement tiled matrix multiplication in Triton with a two-dimensional grid, float32 accumulation, and tail masks. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-matmul |
| Autotuned Matrix Multiplication | Autotune Triton matrix multiplication across tile and pipeline configurations while preserving masked boundary handling. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-matmul-autotune |
| Vector Max Reduction | Compute a vector maximum with one Triton reduction program and masked tail lanes that cannot win comparisons. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-max |
| Single-Pass Mean and Variance | Compute population mean and variance in Triton with single-pass statistics, atomic partials, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-mean-variance |
| ReLU | Implement ReLU activation in Triton with contiguous program tiles, branch-free rectification, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-relu |
| RMSNorm Forward | Implement RMSNorm forward in Triton with per-row square reduction, numerical stability, scaling, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-rmsnorm |
| SiLU | Implement fused SiLU or Swish activation in Triton with contiguous tiles, sigmoid weighting, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-silu |
| Vector Sum Reduction | Implement tiled vector sum reduction in Triton with register partials, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-sum |
| Tiled Transpose | Implement tiled matrix transpose in Triton by swapping load and store strides with masked boundary tiles. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-transpose |
| Vector Addition | Implement elementwise vector addition in Triton with contiguous program tiles and safe masking for partial tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-vector-addition |
| Vectorized Vector Add | Implement vector addition in Triton with larger per-program tiles to reduce launch overhead while masking the final tail. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-vectorized-load |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/thinhphan130997)
<!-- tensortonic:end -->
