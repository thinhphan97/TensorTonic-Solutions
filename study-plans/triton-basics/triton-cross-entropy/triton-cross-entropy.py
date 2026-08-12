import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, loss_out_ptr,
    stride_logits_row,
    B, C,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    valid = cols < C

    row_ptr = logits_ptr + pid * stride_logits_row

    logits = tl.load(
        row_ptr + cols,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    target = tl.load(target_ptr + pid)

    row_max = tl.max(logits, axis=0)

    sum_exp = tl.sum(
        tl.exp(logits - row_max),
        axis=0,
    )

    logsumexp = row_max + tl.log(sum_exp)

    target_logit = tl.sum(
        tl.where(cols == target, logits, 0.0),
        axis=0,
    )

    loss = logsumexp - target_logit

    tl.atomic_add(loss_out_ptr, loss)


def solve(logits: torch.Tensor, target: torch.Tensor, loss_out: torch.Tensor) -> None:
    """Launch the cross-entropy kernel: one program per row, atomic accumulate, then divide by B."""
    B, C = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(C)
    loss_out.zero_()
    grid = (B,)
    cross_entropy_kernel[grid](
        logits, target, loss_out,
        logits.stride(0),
        B, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    loss_out.div_(B)