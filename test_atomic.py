
import torch

import triton
import triton.language as tl


@triton.jit
def atomic_kernel(x_ptr):
    tl.atomic_add(x_ptr, 1)

def test_atomic(x: torch.Tensor, num_tiles:int):
    grid = lambda meta: (num_tiles, )
    atomic_kernel[grid](x)
    return x

def test_correctness(size):
    x = torch.zeros(1, device='cuda')
    y = test_atomic(x, size)
    assert y[0] == size

test_correctness(1)
test_correctness(10)
test_correctness(100)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[128*i for i in range(1, 28, 1)],  # Different possible values for `x_name`.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton'],  # Possible values for `line_arg`.
        line_names=['Triton'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='us',  # Label name for the y-axis.
        plot_name='atomic-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.zeros(1, device='cuda', dtype=torch.int32)
    quantiles = [0.5, 0.2, 0.8]
    # if provider == 'torch':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: test_atomic(x, size), quantiles=quantiles)
    gbps = lambda ms: round(ms * 1000.0, 3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True, save_path='.')

