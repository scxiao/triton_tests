
import torch

import triton
import triton.language as tl


@triton.jit
def atomic_kernel(x_ptr,
                  val,
                  BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + offsets
    v = tl.zeros([BLOCK_SIZE], dtype=tl.float16) + 1
    tl.atomic_add(x_ptrs, v)


def test_atomic(x: torch.Tensor, val: torch.Tensor, num_tiles:int, BLOCK_SIZE):
    grid = lambda meta: (num_tiles, )
    atomic_kernel[grid](x, val, BLOCK_SIZE)


def test_correctness(num_tiles):
    tile_size = 512
    x = torch.zeros((tile_size,), dtype=torch.half, device='cuda')
    val = torch.ones((tile_size,), dtype=torch.half, device='cuda') 

    BLOCK_SIZE=512
    test_atomic(x, val, num_tiles, BLOCK_SIZE)

    # print(f"x = {x}")

    ref = torch.zeros((tile_size, ), dtype = torch.float, device='cuda') + num_tiles
    assert torch.allclose(x.to(torch.float), ref)


test_correctness(3)


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
    tile_size = 512
    BLOCK_SIZE = tile_size
    x = torch.zeros((tile_size), device='cuda', dtype=torch.float16)
    val = x + 1
    quantiles = [0.5, 0.2, 0.8]
    # if provider == 'torch':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: test_atomic(x, val, size, BLOCK_SIZE), quantiles=quantiles)
    gbps = lambda ms: round(ms * 1000.0, 3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True, save_path='.')
