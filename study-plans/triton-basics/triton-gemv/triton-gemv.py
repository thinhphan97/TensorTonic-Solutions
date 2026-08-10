import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        mask_n = cols < N

        a_ptrs = (
            a_ptr
            + rows[:, None] * stride_am
            + cols[None, :] * stride_an
        )

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )

        x = tl.load(
            x_ptr + cols,
            mask=mask_n,
            other=0.0,
        )

        acc += tl.sum(a * x[None, :], axis=1)

    tl.store(
        out_ptr + rows,
        acc,
        mask=mask_m,
    )


def solve(A: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gemv_kernel: out = A @ x."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    gemv_kernel[grid](
        A, x, out,
        M, N,
        A.stride(0), A.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )