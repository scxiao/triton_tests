import torch

import triton  # @manual=//triton:tritonn
import triton.language as tl  # @manual=//triton:tritonl


def atomic_ref(x, v):
    v_max = torch.max(v)
    x[0] = v_max


@triton.jit
def atomic_kernel(x_ptr, 
                  v_ptr,
                  block_n: tl.constexpr):
    offs_n = tl.arange(0, block_n)
    v = tl.load(v_ptr + offs_n)
    v_max = tl.max(v)
    tl.atomic_add(x_ptr, v_max, sem='relaxed')


def run_kernel(x, v):
    N = v.numel()

    def grid(META):
        return [(triton.cdiv(N, META['block_n']))]
    atomic_kernel[grid](x, v, block_n = 256)


def test_correctness(size):

    x = torch.zeros((1,), device='cuda')
    v = torch.randn((size,), device='cuda')

    run_kernel(x, v)

    x_ref = torch.zeros_like(x)
    atomic_ref(x_ref, v)

    print(f"x = {x}")
    print(f"x_ref = {x_ref}")

    torch.testing.assert_close(x, x_ref)

test_correctness(256)
