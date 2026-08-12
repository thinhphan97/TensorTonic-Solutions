import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_bwd_kernel(
    x_ptr, gamma_ptr, dy_ptr,
    dx_ptr, dgamma_ptr, dbeta_ptr,
    stride_x_row, stride_dy_row, stride_dx_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_row_ptr = x_ptr + pid * stride_x_row
    dy_row_ptr = dy_ptr + pid * stride_dy_row
    dx_row_ptr = dx_ptr + pid * stride_dx_row

    x = tl.load(
        x_row_ptr + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # -------------------------
    # Forward statistics
    # -------------------------

    mean = tl.sum(x, axis=0) / N

    x_centered = tl.where(
        mask,
        x - mean,
        0.0,
    )

    var = tl.sum(
        x_centered * x_centered,
        axis=0,
    ) / N

    rstd = tl.rsqrt(var + eps)


    xhat = x_centered * rstd

    # -------------------------
    # Load dy, gamma
    # -------------------------

    dy = tl.load(
        dy_row_ptr + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    gamma = tl.load(
        gamma_ptr + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)


    dxhat = dy * gamma

    # -------------------------
    # dx
    # -------------------------

    sum_dxhat = tl.sum(
        tl.where(mask, dxhat, 0.0),
        axis=0,
    )

    sum_dxhat_xhat = tl.sum(
        tl.where(mask, dxhat * xhat, 0.0),
        axis=0,
    )

    dx = (
        rstd / N
        * (
            N * dxhat
            - sum_dxhat
            - xhat * sum_dxhat_xhat
        )
    )

    tl.store(
        dx_row_ptr + cols,
        dx,
        mask=mask,
    )

    # -------------------------
    # dgamma, dbeta
    # -------------------------

    tl.atomic_add(
        dgamma_ptr + cols,
        dy * xhat,
        mask=mask,
    )

    tl.atomic_add(
        dbeta_ptr + cols,
        dy,
        mask=mask,
    )


def solve(
    x: torch.Tensor, gamma: torch.Tensor, dy: torch.Tensor,
    dx_out: torch.Tensor, dgamma_out: torch.Tensor, dbeta_out: torch.Tensor,
    eps: float,
) -> None:
    """Launch the LayerNorm backward kernel: one program per row, atomic reductions for dgamma and dbeta."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    dgamma_out.zero_()
    dbeta_out.zero_()
    grid = (M,)
    layernorm_bwd_kernel[grid](
        x, gamma, dy,
        dx_out, dgamma_out, dbeta_out,
        x.stride(0), dy.stride(0), dx_out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )