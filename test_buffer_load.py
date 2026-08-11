from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


@gluon.jit
def gluon_buffer_load_to_shared(input: torch.Tensor, 
                            out0: torch.Tensor,
                            out1: torch.Tensor, 
                            n: int,
                            wn: int,
                            BLOCK_SIZE: ttgl.constexpr):
    pid = ttgl.program_id(0)
    
    blocked: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[2], threads_per_warp=[64], warps_per_cta=[4], order=[0])
    shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order = [0])
    padded_shared: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
        [[512, 16]],
        [BLOCK_SIZE],
        [0],
    )
    smem = ttgl.allocate_shared_memory(input.dtype.element_ty, [BLOCK_SIZE], padded_shared)
    offsets = ttgl.arange(0, BLOCK_SIZE, layout=blocked) + pid * BLOCK_SIZE
    masks = offsets < n

    offsets_write = ttgl.arange(0, BLOCK_SIZE, layout=blocked) + pid * BLOCK_SIZE
    mask_write = offsets_write < wn

    gv = ttgl.amd.cdna3.buffer_load(ptr=input, offsets=offsets, mask=masks, other=1.0)
    smem.store(gv)git
    sv = smem.load(layout=blocked)
    tl.store(out0 + offsets_write, sv, mask=mask_write)
    
    ttgl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem, input, offsets, mask=masks, other=0.0
    )
    ttgl.amd.cdna4.async_copy.commit_group()
    ttgl.amd.cdna4.async_copy.wait_group(0)
    sv = smem.load(layout=blocked)
    tl.store(out1 + offsets_write, sv, mask=mask_write)


def test_buffer_load_to_lds(size: int):
    torch.manual_seed(0)
    x = torch.rand(size, dtype=torch.bfloat16, device='cuda')
    y0 = torch.rand(size, dtype=torch.bfloat16, device='cuda')
    y1 = torch.rand(size, dtype=torch.bfloat16, device='cuda')
    print(f"BeforeKernelCall\n")
    print(f"x = {x[-16:-1]}")
    print(f"y0 = {y0[-16:-1]}")
    print(f"y1 = {y1[-16:-1]}")

    n = size - 16
    wn = size
    # grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
    grid = lambda meta: (triton.cdiv(wn, meta['BLOCK_SIZE']), )

    BLOCK_SIZE = 512
    gluon_buffer_load_to_shared[grid](x, y0, y1, n, wn, BLOCK_SIZE)

    print(f"BufferLoad, AfterKernelCall\n")
    print(f"x = {x[-16:-1]}")
    print(f"y0 = {y0[-16:-1]}")
    print(f"y1 = {y1[-16:-1]}")

    # torch.testing.assert_close(x[0:-16], y0[0:-16])
    # torch.testing.assert_close(x[0:-16], y1[0:-16])


@gluon.jit
def gluon_global_load_to_shared(input: torch.Tensor, 
                            out0: torch.Tensor,
                            out1: torch.Tensor, 
                            n: int,
                            wn: int,
                            BLOCK_SIZE: ttgl.constexpr):
    pid = ttgl.program_id(0)
    
    blocked: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[2], threads_per_warp=[64], warps_per_cta=[4], order=[0])
    shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order = [0])
    padded_shared: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
        [[512, 16]],
        [BLOCK_SIZE],
        [0],
    )
    smem = ttgl.allocate_shared_memory(input.dtype.element_ty, [BLOCK_SIZE], padded_shared)
    offsets = ttgl.arange(0, BLOCK_SIZE, layout=blocked) + pid * BLOCK_SIZE
    masks = offsets < n

    offsets_write = ttgl.arange(0, BLOCK_SIZE, layout=blocked) + pid * BLOCK_SIZE
    mask_write = offsets_write < wn

    gv = ttgl.amd.cdna4.buffer_load(ptr=input, offsets=offsets, mask=masks, other=1.0)
    smem.store(gv)
    sv = smem.load(layout=blocked)
    # ttgl.amd.cdna3.buffer_store(ptr=out0, offsets=offsets_write, stored_value=sv, mask=mask_write)
    tl.store(out0 + offsets_write, sv, mask=mask_write)
    
    ttgl.amd.cdna4.async_copy.global_load_to_shared(
        smem, input + offsets, mask=masks, other=0.0
    )
    ttgl.amd.cdna4.async_copy.commit_group()
    ttgl.amd.cdna4.async_copy.wait_group(0)
    sv = smem.load(layout=blocked)
    tl.store(out1 + offsets_write, sv, mask=mask_write)


def test_global_load_to_lds(size: int):
    torch.manual_seed(0)
    x = torch.rand(size, dtype=torch.bfloat16, device='cuda')
    y0 = torch.rand(size, dtype=torch.bfloat16, device='cuda')
    y1 = torch.rand(size, dtype=torch.bfloat16, device='cuda')
    print(f"GlobalLoad, BeforeKernelCall\n")
    print(f"x = {x[-16:-1]}")
    print(f"y0 = {y0[-16:-1]}")
    print(f"y1 = {y1[-16:-1]}")

    n = size - 16
    wn = size
    # grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
    grid = lambda meta: (triton.cdiv(wn, meta['BLOCK_SIZE']), )

    BLOCK_SIZE = 512
    gluon_global_load_to_shared[grid](x, y0, y1, n, wn, BLOCK_SIZE)

    print(f"GlobalLoad, AfterKernelCall\n")
    print(f"x = {x[-16:-1]}")
    print(f"y0 = {y0[-16:-1]}")
    print(f"y1 = {y1[-16:-1]}")

    # torch.testing.assert_close(x[0:-16], y0[0:-16])
    # torch.testing.assert_close(x[0:-16], y1[0:-16])

def main():
    test_buffer_load_to_lds(1024 * 1024)
    test_global_load_to_lds(1024 * 1024)


if __name__ == "__main__":
    main()
