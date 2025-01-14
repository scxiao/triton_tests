import numpy as np
import torch
import triton
import triton.language as tl
import re
import pytest

#This version is based on version 5 contains peel off last iteration

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


class TorchGemm(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, a, b):
        x = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        return x.to(torch.half)

nstages=2
 
def _get_tune_configs():
    configs = [
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=8),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=8),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=8),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=8),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE_M': 4, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=8),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=8),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=8),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=8),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=4),

        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=4),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=nstages, num_warps=4),

        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 512, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 464, 'kpack': 2}, num_stages=nstages, num_warps=2),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 2048, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=nstages, num_warps=2),
    ] if is_hip() else [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1}, num_stages=2, num_warps=4),
    ]
    return configs
 
 
# @triton.autotune(
#     configs=_get_tune_configs(),
#     key=['M', 'N', 'K'],
# )
# @triton.heuristics({
#     'EVEN_K': lambda args: args['K'] % (args['BLOCK_K']) == 0,
# })
@triton.jit
def _triton_gemm_kernel(
    # Pointers to matrices
    A,
    B,
    C,
    # Matrix dimensions
    M,
    stride_am,
    stride_ak,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_M)
    a_ptrs = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = B + (rn[None, :] * stride_ak + rk[:, None] * stride_am)
    # Write back the block of the output matrix C with masks.
    c_ptrs = C + stride_am * rm[:, None] + rn[None, :]
    c_mask = (rm[:, None] < M) & (rn[None, :] < M)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # _0 = tl.zeros([1, 1], dtype=A.dtype.element_ty)
    acc_type = tl.int32 if A.dtype.element_ty == tl.int8 else tl.float32
    accumulator = tl.zeros([BLOCK_M, BLOCK_M], dtype=acc_type)
    loop_k = tl.cdiv(M, BLOCK_M)

    for _ in range(0, loop_k):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_M * stride_ak
        b_ptrs += BLOCK_M * stride_am
        tl.atomic_add(c_ptrs, accumulator, c_mask)

    c = accumulator.to(C.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)

 
def gemm_forward(out, a, b):
    # Check constraints.
    assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    assert out.dtype == torch.float16 or out.dtype == torch.bfloat16, "Output type must be float16 or bfloat16"
    # assert a.shape[1] == b.shape[1], "Matrix B must be transposed"
    M, K = a.shape
    K, N = b.shape
 
    kwargs = [
        a,
        b,
        out,
        M,
        a.stride(0),
        a.stride(1),
        32,
    ]
 
    _triton_gemm_kernel[(1, )](*kwargs, num_warps=4, num_stages=4)
 

def get_shapes():
    shapes = [(32, 32, 128)]

    return shapes


TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')
tl_to_torch_types = {
    tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16,
    tl.float32: torch.float32,
    tl.int8: torch.int8,
    tl.int32: torch.int32,
}
if TORCH_HAS_FP8E5B16:
    tl_to_torch_types[tl.float8e5b16] = torch.float8_e5m2fnuz
if TORCH_HAS_FP8E4B8:
    tl_to_torch_types[tl.float8e4b8] = torch.float8_e4m3fnuz

name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8e4': tl.float8e4b8,
    'fp8e5': tl.float8e5b16,
}

def gen_input(M, N, ty_name, needTrans, seed, device='cuda'):
    d_type = name_to_tl_types[ty_name]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    if ty_name == 'int8':
        if needTrans:
            raw_data = torch.randint(-20, 20, (N, M), dtype=torch.int8, device='cuda').T
        else:
            raw_data = torch.randint(-20, 20, (M, N), dtype=torch.int8, device='cuda')

        return raw_data, raw_data.to(torch.half)

    if needTrans:
        raw_data = torch.randn((N, M), dtype=torch.float32, device='cuda').T
    else:
        raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda')
    # avoid type conversion rounding errors of subnormal values
    raw_data += 0.1
    if d_type == tl.float8e4b8:
        raw_data += torch.sign(raw_data)

    if (d_type == tl.float8e4b8 and TORCH_HAS_FP8E4B8) or \
        (d_type == tl.float8e5b16 and TORCH_HAS_FP8E5B16) or not d_type.is_fp8():
        input = raw_data.to(tl_to_torch_types[d_type])
        input_f16 = input.to(torch.float16)
    else:
        f8_tensor = raw_data.to(torch.int8)
        # keep only two bits of exponent to avoid overflow
        f8_tensor = f8_tensor & 0b00111111
        input = triton.reinterpret(f8_tensor, d_type)
        input_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        n_elements = raw_data.numel()
        copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)

    return input, input_f16


def get_type(provider):
    res = re.findall(r'\(.*?\)', provider)
    return res[0][1:-1]

def num_tensors(M, N, K):
    size = M * N + M * K + N * K + M + N
    total_size = 512 * 1024 * 1024
    num = triton.cdiv(total_size, size)
    return num 


# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=get_shapes(),
        line_arg='provider',
        # line_vals=['triton(int8)', 'triton(fp8e4)', 'triton(fp8e5)', 'torch(int8)'],
        # line_names=['Triton.int8', 'Triton.fp8e4', 'Triton.fp8e5', "Torch.int8"],
        line_vals=['triton(fp16)', 'torch(fp16)'],
        line_names=['Triton.fp16', "Torch.fp16"],
        # styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        args={},
        plot_name='gemm-perf',
    )
)
def benchmark(M, N, K, provider):
    in_dtype = get_type(provider)
    out_dtype = torch.half

    a, _ = gen_input(M, K, in_dtype, False, 1, device='cuda')
    b, _ = gen_input(K, N, in_dtype, True, 2, device='cuda')
    out = torch.empty([M, N], dtype=torch.half, device='cuda')

    quantiles = [0.5, 0.2, 0.8]
 
    if 'torch' in provider:
        torch_gemm = TorchGemm()
        # ms, min_ms, max_ms = triton.testing.do_bench_rotating_tensor(
        #     lambda i: torch_gemm(a[i % tensor_num], b[i % tensor_num]), rep=100, quantiles=quantiles
        # )
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_gemm(a, b), rep=100, quantiles=quantiles
        )
    else: 
        assert 'triton' in provider
        # out = torch.empty([a.shape[0], b.shape[1]], dtype=torch.half, device=a.device)
        # ms, min_ms, max_ms = triton.testing.do_bench_rotating_tensor(
        #     lambda i: gemm_forward(out[i % tensor_num], a[i % tensor_num], b[i % tensor_num]), rep=100, quantiles=quantiles
        # )
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: gemm_forward(out, a, b), rep=100, quantiles=quantiles
        )
        # print(f"M = {M}, N = {N}, K = {K}, type = {in_dtype}, best_config = {_triton_gemm_kernel.best_config}")
        # print(f'GEMM SIZE: {M},{N},{K} Best tuning config: ({_triton_gemm_kernel.get_best_config()})')
        # print(f'GEMM SIZE: {M},{N},{K} TIME: {ms:.3f} ms, {min_ms:.3f} min_ms, {max_ms:.3f} max_ms')
    perf_us = lambda x: round(x * 1e3, 2)
    # perf_us = lambda x: round(2 * M * N * K / x * 1e-9, 2)
    return perf_us(ms), perf_us(min_ms), perf_us(max_ms)
 
 
if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)


@pytest.mark.parametrize('m, n, k', get_shapes())
def test_gemm(m, n, k):
    torch.random.manual_seed(0)
    with torch.no_grad():
        # a = torch.randint(-12, 12, (m, k), dtype=torch.int8).cuda()
        # b = torch.randint(-12, 12, (n, k), dtype=torch.int8).cuda().T

        a, _ = gen_input(m, k, 'int8', False, 1, device='cuda')
        b, _ = gen_input(k, n, 'int8', True, 2, device='cuda')

        torch_gemm = TorchGemm()
        out_torch = torch_gemm(a, b)
        out_triton = torch.empty([a.shape[0], b.shape[1]], dtype=torch.half, device=a.device)
        gemm_forward(out_triton, a, b)
        print(f"M = {m}, N = {n}, K = {k}, best_config = {_triton_gemm_kernel.best_config}")

        print(f"out_torch = {out_torch}")
        print(f"out_triton = {out_triton}")

        diff = ~np.isclose(out_triton.half().cpu().numpy(), out_torch.half().cpu().numpy(), rtol=1e-2)
        assert diff.sum() < 10, f"m={m}, n={n}, k={k}"
