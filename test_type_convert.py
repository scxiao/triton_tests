import torch

import triton
import triton.language as tl

import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ================================================
# type conversion kernel for x.dtype to y.dtype
@triton.jit
def fp32_to_bf16(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr,):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    out = x.to(y_ptr.dtype.element_ty)
    y = tl.store(y_ptr + offs, out)


def type_conversion(x, y):
    n_elem = x.numel()
    BLOCK_SIZE=1024
    grid = (triton.cdiv(n_elem, BLOCK_SIZE),)
    fp32_to_bf16[grid](x, y, BLOCK_SIZE)

def test_type_conversion():
    torch.manual_seed(0)
    # M, N = 222, 213
    N = 1024 * 304
    x = torch.rand(N, dtype=torch.float32, device=DEVICE)
    a = torch.tensor(0x86db8000, dtype=torch.uint32).view(torch.float32)
    x[0] = a
    y = torch.empty_like(x, dtype=torch.bfloat16, device=DEVICE)
    type_conversion(x, y)

    print(f"x = {x}")
    print(f"x_bf16 = {x.to(torch.bfloat16)}")
    print(f"y = {y}")

    print(f"x = {hex(x[0].view(torch.uint32))}")
    print(f"y = {hex(y[0].to(torch.float32).view(torch.uint32))}")

    torch.testing.assert_close(y, x.to(torch.bfloat16))

test_type_conversion()

# # run perf of different kernels
# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=['N'],  # Argument names to use as an x-axis for the plot.
#         x_vals=[2**i for i in range(12, 24, 1)],  # Different possible values for `x_name`.
#         x_log=True,  # x axis is logarithmic.
#         line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
#         line_vals=['trans_op_block_64', 'torch'],  # Possible values for `line_arg`.
#         line_names=['Trans_op_block_64', 'Torch'],  # Label name for the lines.
#         # line_vals=['trans_block_32', 'trans_block_64', 'trans_op_block_64', 'torch'],  # Possible values for `line_arg`.
#         # line_names=['Trans_block_32', 'Trans_block_64', 'Trans_op_block_64', 'Torch'],  # Label name for the lines.
#         # styles=[('blue', '-'), ('green', '-')],  # Line styles.
#         ylabel='GB/s',  # Label name for the y-axis.
#         plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
#         args={},  # Values for function arguments not in `x_names` and `y_name`.
#     ))
# def benchmark(N, provider):
#     M = 256
#     x = torch.rand((M, N), device=DEVICE, dtype=torch.float32)
#     out =  torch.empty_like(x)
#     quantiles = [0.5, 0.2, 0.8]
#     ms, max_ms, min_ms = 1, 1, 1
#     # if provider == 'triton':
#     #     y = torch.rand((M, N), device=DEVICE, dtype=torch.float32)
#     #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: add2d_op(x, y, out), quantiles=quantiles)
#     # if provider == 'trans_block_32':
#     #     y = torch.rand((N, M), device=DEVICE, dtype=torch.float32).T
#     #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: add2d_trans_input_op(x, y, out), quantiles=quantiles)
#     # if provider == 'trans_block_64':
#     #     y = torch.rand((N, M), device=DEVICE, dtype=torch.float32).T
#     #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: add2d_trans_input_op1(x, y, out), quantiles=quantiles)
#     if provider == 'trans_op_block_64':
#         y = torch.rand((N, M), device=DEVICE, dtype=torch.float32)
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: add2d_trans_op(x, y, out), quantiles=quantiles)
#     if provider == 'torch':
#         y = torch.rand((M, N), device=DEVICE, dtype=torch.float32)
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
#     gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
#     return gbps(ms), gbps(max_ms), gbps(min_ms)


# # %%
# # We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# # `save_path='/path/to/results/' to save them to disk along with raw CSV data:
# # benchmark.run(print_data=True, show_plots=True)
