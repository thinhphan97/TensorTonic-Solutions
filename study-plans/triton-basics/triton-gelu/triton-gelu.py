import torch
import triton
import triton.language as tl

SQRT_2: tl.constexpr = 1.4142135623730951


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    sqrt_2 = tl.sqrt(2.0)

    tl.store(
        out_ptr + offsets, 
        0.5 * x * (1 + tl.erf(x / SQRT_2)), 
        mask=mask
    )

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gelu_kernel: out = 0.5 * x * (1 + erf(x / sqrt(2)))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)