import torch

import triton  # @manual=//triton:tritonn
import triton.language as tl  # @manual=//triton:tritonl


def torch_atomic_add(x_ptr, idx_ptr, v_ptr, row_size, column_size, block_m, block_n):
    for r in range(row_size):
        for i in range(block_m):
            for j in range(block_n):
                # if j % 2 == 0:
                x_ptr[r][idx_ptr[i][j]] += v_ptr[i][j]


@triton.jit
def atomic_kernel(x_ptr, 
                  idx_ptr, 
                  v_ptr,
                  row_size,
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

    # mask_n = offs_n % 2 == 0

    # tl.atomic_add(x_ptrs + idx, val, mask = mask_n[None, :], sem="relaxed")
    tl.atomic_add(x_ptrs + idx, val, sem="relaxed")


def run_once(x, idx, val, block_m, block_n):
    tile_num = x.shape[0]
    atomic_kernel[(tile_num,)](x, idx, val, x.shape[0], x.shape[1], block_m, block_n)


def check_correctness(row_num, column_num, block_m=32, block_n=32, dtype=torch.float32, trans = False):
    x = torch.zeros((row_num, column_num), dtype=dtype, device="cuda")
    x_ref = torch.zeros((row_num, column_num), dtype=dtype, device="cuda")

    i_val = list(range(column_num))
    idx = torch.tensor(i_val, dtype=torch.int32, device='cuda')
    idx = idx.broadcast_to(block_m // 2, column_num).contiguous()
    idx = idx.reshape((-1, block_n))
    if trans:
        idx = idx.transpose(1, 0).contiguous()

    # idx = torch.randint(0, column_num, (block_m, block_n), dtype=torch.int32, device='cuda')
    val = torch.rand((block_m, block_n), dtype=dtype, device='cuda')
    run_once(x, idx, val, block_m, block_n)
    torch_atomic_add(x_ref, idx, val, x_ref.shape[0], x_ref.shape[1], block_m, block_n)

    # print(f"x = {x}")
    # print(f"x_ref = {x_ref}")

    torch.testing.assert_close(x, x_ref)

def test_trans():
    check_correctness(10, 64, trans=True)

def test_no_trans():
    check_correctness(10, 64, trans=True)

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
    column_num = 64
    block_m = 32
    block_n = 32
    x = torch.zeros((row_num, column_num), dtype=dtype, device="cuda")
    i_val = list(range(column_num))
    idx = torch.tensor(i_val, dtype=torch.int32, device='cuda')
    idx = idx.broadcast_to(block_m // 2, column_num).contiguous()
    idx = idx.reshape((-1, block_n))
    idx = idx.transpose(1, 0).contiguous()
    # print(f"idx_shape = {idx.shape}")

    # idx = torch.randint(0, column_num, (block_m, block_n), dtype=torch.int32, device='cuda')
    val = torch.rand((block_m, block_n), dtype=dtype, device='cuda')
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: run_once(x, idx, val, block_m, block_n), quantiles=quantiles
    )
    ms2us = lambda ms: round(ms * 1000.0, 3)
    return ms2us(ms), ms2us(max_ms), ms2us(min_ms)


def main():
    benchmark.run(print_data=True, show_plots=True, save_path=".")

if __name__ == "__main__":
    main()

