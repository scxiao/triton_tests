import torch

import triton
import triton.language as tl

import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()

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


def add2d_trans_op(x, y, output):
    M, N = output.shape
    BLOCK_M = 32
    BLOCK_N = 128
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    kernel_add2d_trans_op[grid](x, y, output, 
                      M, N, 
                      x.stride(0), x.stride(1), 
                      y.stride(0), y.stride(1), 
                      output.stride(0), output.stride(1), 
                      BLOCK_M, BLOCK_N, num_warps=16)

    return output

def test_add2d_trans_op():
    torch.manual_seed(0)
    # M, N = 222, 213
    M, N = 12800, 2560
    x = torch.rand((M, N), device=DEVICE)
    y = torch.rand((N, M), device=DEVICE)
    output_triton = torch.empty_like(x)
    output_torch = x + y.T
    add2d_trans_op(x, y, output_triton)

    torch.testing.assert_close(output_triton, output_torch)

test_add2d_trans_op()

# run perf of different kernels
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 24, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['trans_op_block_64', 'torch'],  # Possible values for `line_arg`.
        line_names=['Trans_op_block_64', 'Torch'],  # Label name for the lines.
        # line_vals=['trans_block_32', 'trans_block_64', 'trans_op_block_64', 'torch'],  # Possible values for `line_arg`.
        # line_names=['Trans_block_32', 'Trans_block_64', 'Trans_op_block_64', 'Torch'],  # Label name for the lines.
        # styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(N, provider):
    M = 256
    x = torch.rand((M, N), device=DEVICE, dtype=torch.float32)
    out =  torch.empty_like(x)
    quantiles = [0.5, 0.2, 0.8]
    ms, max_ms, min_ms = 1, 1, 1
    # if provider == 'triton':
    #     y = torch.rand((M, N), device=DEVICE, dtype=torch.float32)
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: add2d_op(x, y, out), quantiles=quantiles)
    # if provider == 'trans_block_32':
    #     y = torch.rand((N, M), device=DEVICE, dtype=torch.float32).T
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: add2d_trans_input_op(x, y, out), quantiles=quantiles)
    # if provider == 'trans_block_64':
    #     y = torch.rand((N, M), device=DEVICE, dtype=torch.float32).T
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: add2d_trans_input_op1(x, y, out), quantiles=quantiles)
    if provider == 'trans_op_block_64':
        y = torch.rand((N, M), device=DEVICE, dtype=torch.float32)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add2d_trans_op(x, y, out), quantiles=quantiles)
    if provider == 'torch':
        y = torch.rand((M, N), device=DEVICE, dtype=torch.float32)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
# benchmark.run(print_data=True, show_plots=True)
