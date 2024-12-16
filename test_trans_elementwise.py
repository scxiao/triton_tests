import torch

import triton
import triton.language as tl

import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ================================================
# 1d vector add
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


# ================================================
# 2d tensor add
@triton.jit
def kernel_add2d(x_ptr, y_ptr, out_ptr, M, N, stride_m, stride_n, BLOCK_M:tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = off_m[:, None] * stride_m + off_n[None, :] * stride_n
    mask = off_m[:, None] < M and off_n[None, :] < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


def add2d_op(x, y):
    output = torch.empty_like(x)
    M, N = output.shape
    BLOCK_M = 32
    BLOCK_N = 32
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    kernel_add2d[grid](x, y, output, M, N, output.stride(0), output.stride(1), BLOCK_M, BLOCK_N)

    return output


def test_add2d():
    torch.manual_seed(0)
    M, N = 222, 213
    x = torch.rand((M, N), device=DEVICE)
    y = torch.rand((M, N), device=DEVICE)
    output_torch = x + y
    output_triton = add2d_op(x, y)
    print(output_torch)
    print(output_triton)

    torch.testing.assert_close(output_triton, output_torch)


# ================================================
# 2d add with one operand to be transpose
@triton.jit
def kernel_add2d_trans_input(x_ptr, y_ptr, out_ptr, 
          M, N, 
          stride_xm, stride_xn, 
          stride_ym, stride_yn, 
          stride_om, stride_on, 
          BLOCK_M: tl.constexpr, 
          BLOCK_N: tl.constexpr,):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_x = off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
    offsets_y = off_m[:, None] * stride_ym + off_n[None, :] * stride_yn
    offsets_out = off_m[:, None] * stride_om + off_n[None, :] * stride_on
    mask = off_m[:, None] < M and off_n[None, :] < N
    x = tl.load(x_ptr + offsets_x, mask=mask)
    y = tl.load(y_ptr + offsets_y, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets_out, output, mask=mask)


def add2d_trans_input_op(x, y):
    output = torch.empty_like(x)
    M, N = output.shape
    BLOCK_M = 32
    BLOCK_N = 32
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    kernel_add2d_trans_input[grid](x, y, output, 
                      M, N, 
                      x.stride(0), x.stride(1), 
                      y.stride(0), y.stride(1), 
                      output.stride(0), output.stride(1), 
                      BLOCK_M, BLOCK_N)

    return output

def test_add2d_trans_input():
    torch.manual_seed(0)
    M, N = 222, 213
    x = torch.rand((N, M), device=DEVICE).T
    y = torch.rand((M, N), device=DEVICE)
    output_torch = x + y
    output_triton = add2d_trans_input_op(x, y)
    print(output_torch)
    print(output_triton)

    torch.testing.assert_close(output_triton, output_torch)



# ================================================
# 2d add with transposed input
@triton.jit
def kernel_add2d_trans_input1(x_ptr, y_ptr, out_ptr, 
          M, N, 
          stride_xm, stride_xn, 
          stride_ym, stride_yn, 
          stride_om, stride_on, 
          BLOCK_M: tl.constexpr, 
          BLOCK_N: tl.constexpr,):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_x = off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
    offsets_y = off_m[:, None] * stride_ym + off_n[None, :] * stride_yn
    offsets_out = off_m[:, None] * stride_om + off_n[None, :] * stride_on
    mask = off_m[:, None] < M and off_n[None, :] < N
    x = tl.load(x_ptr + offsets_x, mask=mask)
    y = tl.load(y_ptr + offsets_y, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets_out, output, mask=mask)


def add2d_trans_input_op1(x, y):
    output = torch.empty_like(x)
    M, N = output.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    kernel_add2d_trans_input1[grid](x, y, output, 
                      M, N, 
                      x.stride(0), x.stride(1), 
                      y.stride(0), y.stride(1), 
                      output.stride(0), output.stride(1), 
                      BLOCK_M, BLOCK_N)

    return output

def test_add2d_trans_input1():
    torch.manual_seed(0)
    M, N = 222, 213
    x = torch.rand((N, M), device=DEVICE).T
    y = torch.rand((M, N), device=DEVICE)
    output_torch = x + y
    output_triton = add2d_trans_input_op1(x, y)
    print(output_torch)
    print(output_triton)

    torch.testing.assert_close(output_triton, output_torch)


# ================================================
# 2d add with tran_op
@triton.jit
def kernel_add2d_trans_op(x_ptr, y_ptr, out_ptr, 
          M, N, 
          stride_xm, stride_xn, 
          stride_yn, stride_ym, 
          stride_om, stride_on, 
          BLOCK_M: tl.constexpr, 
          BLOCK_N: tl.constexpr,):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_x = off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
    offsets_y = off_m[None, :] * stride_ym + off_n[:, None] * stride_yn
    offsets_out = off_m[:, None] * stride_om + off_n[None, :] * stride_on
    mask = off_m[:, None] < M and off_n[None, :] < N
    mask_y = off_m[None, :] < M and off_n[:, None] < N
    x = tl.load(x_ptr + offsets_x, mask=mask)
    y = tl.load(y_ptr + offsets_y, mask=mask_y)
    y = tl.trans(y)
    output = x + y
    tl.store(out_ptr + offsets_out, output, mask=mask)


def add2d_trans_op(x, y):
    output = torch.empty_like(x)
    M, N = output.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    kernel_add2d_trans_op[grid](x, y, output, 
                      M, N, 
                      x.stride(0), x.stride(1), 
                      y.stride(0), y.stride(1), 
                      output.stride(0), output.stride(1), 
                      BLOCK_M, BLOCK_N)

    return output

def test_add2d_trans_op():
    torch.manual_seed(0)
    M, N = 222, 213
    x = torch.rand((M, N), device=DEVICE)
    y = torch.rand((N, M), device=DEVICE)
    output_torch = x + y.T
    output_triton = add2d_trans_op(x, y)
    print(output_torch)
    print(output_triton)

    torch.testing.assert_close(output_triton, output_torch)
