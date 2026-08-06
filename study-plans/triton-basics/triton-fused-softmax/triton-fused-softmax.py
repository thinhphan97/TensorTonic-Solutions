import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)

    x_row_ptr = x_ptr + pid * x_row_stride
    out_row_ptr = out_ptr + pid * out_row_stride
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    x = tl.load(x_row_ptr + cols, mask=mask, other=-float('inf'))
    
    row_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - row_max)
    
    x_exp = tl.where(mask, x_exp, 0.0)
    x_sum_exp = tl.sum(x_exp, axis=0)

    out_vals = x_exp / x_sum_exp
    
    tl.store(out_row_ptr + cols, out_vals, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )