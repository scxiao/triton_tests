# this repro has been extracted by running the unit test
import triton
import triton.language as tl

import torch
from torch import tensor
# from torch._inductor.runtime import triton_helpers, triton_heuristics
# from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
# from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
# triton_helpers.set_driver_to_gpu()

# @triton.jit
# def get_dtype_e_bits(dtype):
# #     tl.device_print("dtype = ", dtype)
#     if dtype == tl.float16:
#         nbits: tl.constexpr = 5
#     elif dtype == tl.bfloat16:
#         nbits: tl.constexpr = 8
#     elif dtype == tl.float8e4nv:
#         nbits: tl.constexpr = 4
#     elif dtype == tl.float8e5:
#         nbits: tl.constexpr = 5
#     else:
#         nbits: tl.constexpr = 10
#     return nbits

@triton.jit
def get_dtype_e_bits(dtype):
#     tl.device_print("dtype = ", dtype)
    if dtype is tl.float16:
        nbits: tl.constexpr = 5        
    else:
        nbits: tl.constexpr = 10
    return nbits


@triton.jit
def triton_poi_fused_scatter_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (5*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 4")
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(5*tmp0, [XBLOCK])), tmp2, xmask, sem='relaxed')


inp0, inp1, out, xnumel, XBLOCK = [tensor([[1],
        [2],
        [3]], device='cuda:0'),
 tensor([[-0.1117, -0.4966,  0.1631, -0.8818,  0.2891],
        [ 0.4900, -0.3853, -0.7119,  0.6367, -0.7139],
        [-1.0830, -0.5547, -1.3252,  0.6968, -0.6631],
        [ 1.2158, -2.5273,  1.4775, -0.1697, -0.9917]], device='cuda:0',
       dtype=torch.float16),
 tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]], device='cuda:0', dtype=torch.float16),
 3,
 4]

grid=(1,1,1)

triton_poi_fused_scatter_1[grid](inp0, inp1, out, xnumel, XBLOCK)
