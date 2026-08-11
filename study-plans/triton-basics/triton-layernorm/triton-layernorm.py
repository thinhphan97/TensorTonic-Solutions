import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(
        x_ptr + row * stride_x_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    mean = tl.sum(x, axis=0) / N

    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    x_hat = (x - mean) * rstd

    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)

    y = x_hat * gamma + beta

    tl.store(
        out_ptr + row * stride_out_row + offsets,
        y,
        mask=mask,
    )


def solve(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the LayerNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    layernorm_fwd_kernel[grid](
        x, gamma, beta, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )