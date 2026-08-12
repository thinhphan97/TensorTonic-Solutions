import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr, mask_ptr, out_ptr,
    n, p,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    valid = offsets < n

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0)

    mask = tl.load(mask_ptr + offsets, mask=valid, other=0)

    out = x*mask/(1-p)

    tl.store(out_ptr + offsets, out, mask=valid)


def solve(x: torch.Tensor, mask: torch.Tensor, out: torch.Tensor, p: float) -> None:
    """Launch the dropout kernel: 1D grid over the input vector."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    dropout_kernel[grid](
        x, mask, out,
        n, p,
        BLOCK_SIZE=BLOCK_SIZE,
    )