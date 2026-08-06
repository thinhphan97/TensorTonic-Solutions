import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):

    row_idx = tl.program_id(axis=0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    valid_cols = col_offsets < n_cols

    row_ptrs = x_ptr + row_idx * x_row_stride + col_offsets


    row_values = tl.load(
        row_ptrs,
        mask=valid_cols,
        other=-float("inf"),
    ).to(tl.float32)


    row_max = tl.max(row_values, axis=0)
    shifted_values = row_values - row_max
    exp_sum = tl.sum(tl.exp(shifted_values), axis=0)
    logsumexp = row_max + tl.log(exp_sum)

    tl.store(out_ptr + row_idx, logsumexp)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )