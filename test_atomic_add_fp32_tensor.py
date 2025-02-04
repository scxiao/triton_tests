import torch

import triton  # @manual=//triton:tritonn
import triton.language as tl  # @manual=//triton:tritonl


@triton.jit
def atomic_kernel(x_ptr, 
                  idx_ptr, 
                  v_ptr,
                  column_size,
                  block_m: tl.constexpr, 
                  block_n: tl.constexpr):
    pid = tl.program_id(0)
    x_ptrs = x_ptr + pid * column_size
    offs_m = tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    idx_ptrs = idx_ptr + offs_m[:, None] * block_n + offs_n[None, :]
    v_ptrs = v_ptr + offs_m[:, None] * block_n + offs_n[None, :]
    idx = tl.load(idx_ptrs)
    val = tl.load(v_ptrs)

    tl.atomic_add(x_ptrs + idx, val, sem="relaxed")


def run_once(x, idx, val, block_m, block_n):
    tile_num = x.shape[0]
    atomic_kernel[(tile_num,)](x, idx, val, x.shape[1], block_m, block_n)


def test(row_num, column_num, block_m=32, block_n=32, dtype=torch.float32):
    x = torch.zeros((row_num, column_num), dtype=dtype, device="cuda")
    idx = torch.randint(0, column_num, (block_m, block_n), dtype=torch.int32, device='cuda')
    val = torch.rand((block_m, block_n), dtype=dtype, device='cuda')
    run_once(x, idx, val, block_m, block_n)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["row_num"],
        x_vals=[256 * i for i in range(1, 16, 1)],
        line_arg="dtype",
        line_vals=[torch.float32],
        line_names=["fp32"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="atomicadd",
        args={},
    )
)
def benchmark(row_num, dtype):
    quantiles = [0.5, 0.2, 0.8]
    column_num = 5
    block_m = 32
    block_n = 32
    x = torch.zeros((row_num, column_num), dtype=dtype, device="cuda")
    idx = torch.randint(0, column_num, (block_m, block_n), dtype=torch.int32, device='cuda')
    val = torch.rand((block_m, block_n), dtype=dtype, device='cuda')
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: run_once(x, idx, val, block_m, block_n), quantiles=quantiles
    )
    ms2us = lambda ms: round(ms * 1000.0, 3)
    return ms2us(ms), ms2us(max_ms), ms2us(min_ms)


def main():
    benchmark.run(print_data=True, show_plots=True, save_path=".")
    # test(1, 20, 32, 32)

if __name__ == "__main__":
    main()

