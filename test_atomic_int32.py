
import torch

import triton
import triton.language as tl


@triton.jit
def atomic_kernel(x_ptr, 
                out,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    out_ptrs = out + offsets
    # val = tl.atomic_add(x_ptr, 1, sem='relaxed')
    val = tl.atomic_add(x_ptr, 1)
    tl.store(out_ptrs, val)


def test_atomic(x: torch.Tensor, out: torch.Tensor, num_tiles:int, BLOCK_SIZE: tl.constexpr):
    grid = lambda meta: (num_tiles, )
    atomic_kernel[grid](x, out, BLOCK_SIZE)
    return out

def test_correctness(num_tiles):
    x = torch.zeros(1, device='cuda')
    BLOCK_SIZE=1024
    out = torch.randn((num_tiles * BLOCK_SIZE), dtype=torch.float, device='cuda')
    test_atomic(x, out, num_tiles, BLOCK_SIZE)
    sorted_vals, _ = torch.sort(out)
    out_ref = torch.zeros((num_tiles * BLOCK_SIZE), dtype=torch.float, device='cuda')
    for i in range(num_tiles):
        out_ref[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] = i
    assert torch.allclose(sorted_vals, out_ref)


# test_correctness(1)
# test_correctness(10)
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
    x = torch.zeros(2, device='cuda', dtype=torch.int32)
    quantiles = [0.5, 0.2, 0.8]
    BLOCK_SIZE = 1024
    out = torch.randn((size * BLOCK_SIZE), dtype=torch.float, device='cuda')
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: test_atomic(x, out, size, BLOCK_SIZE), quantiles=quantiles)
    gbps = lambda ms: round(ms * 1000.0, 3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True, save_path='.')

