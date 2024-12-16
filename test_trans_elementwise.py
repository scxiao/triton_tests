import torch

import triton
import triton.language as tl

import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def vec_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


def add_op(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE=1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    vec_add[grid](x, y, output, n_elements, BLOCK_SIZE)

    return output


def test_vec_add():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add_op(x, y)
    print(output_torch)
    print(output_triton)

    torch.testing.assert_close(output_triton, output_torch)

