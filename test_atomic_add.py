
import torch

import triton
import triton.language as tl


@triton.jit
def atomic_kernel(x_ptr, 
                out,
                BLOCK_SIZE: tl.constexpr):
    # x_ptrs = x_ptr + tl.arange(0, 2)
    pid = tl.program_id(0)
    offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    out_ptrs = out + offsets
    # tl.atomic_add(x_ptrs, 1, sem='relaxed')
    val = tl.atomic_add(x_ptr, 1)
    tl.store(out_ptrs, val[None])


def test_atomic(x: torch.Tensor, num_tiles:int, BLOCK_SIZE):
    grid = lambda meta: (num_tiles, )
    out = torch.randn((num_tiles * BLOCK_SIZE), dtype=torch.float, device='cuda')
    atomic_kernel[grid](x, out, BLOCK_SIZE)
    return out


def test_correctness(num_tiles):
    x = torch.zeros(1, device='cuda')
    BLOCK_SIZE=256
    y = test_atomic(x, num_tiles, BLOCK_SIZE)
    for i in range(num_tiles):
        offset = i * BLOCK_SIZE
        print(f"y[{i}] =\n{y[offset:offset + BLOCK_SIZE]}")

test_correctness(3)
