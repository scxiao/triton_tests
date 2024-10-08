import torch

import triton
import triton.language as tl

import pytest
import tempfile


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=0),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c


gemm_ttgir = """
// -----// IR Dump Before ConvertTritonAMDGPUToLLVM (convert-triton-amdgpu-to-llvm) ('builtin.module' operation) //----- //
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":91:0)
#loc1 = loc(unknown)
#loc34 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":152:18)
#loc39 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":151:18)
#loc40 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":143:22)
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [32, 32], isTransposed = false}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 12288 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":91:0), %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":91:0), %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":91:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":91:0), %arg4: i32 {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":91:0), %arg5: i32 {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":91:0), %arg6: i32 {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":91:0), %arg7: i32 {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":91:0), %arg8: i32 {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":91:0)) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<128x16xi32, #blocked> loc(#loc1)
    %c16_i32 = arith.constant 16 : i32 loc(#loc1)
    %c256_i32 = arith.constant 256 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #blocked> loc(#loc1)
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x256xf16, #blocked1> loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c127_i32 = arith.constant 127 : i32 loc(#loc1)
    %c255_i32 = arith.constant 255 : i32 loc(#loc1)
    %c15_i32 = arith.constant 15 : i32 loc(#loc1)
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma> loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.addi %arg3, %c127_i32 : i32 loc(#loc55)
    %2 = arith.divsi %1, %c128_i32 : i32 loc(#loc56)
    %3 = arith.addi %arg4, %c255_i32 : i32 loc(#loc57)
    %4 = arith.divsi %3, %c256_i32 : i32 loc(#loc58)
    %5 = arith.divsi %0, %4 : i32 loc(#loc7)
    %6 = arith.subi %2, %5 : i32 loc(#loc8)
    %7 = arith.minsi %6, %c1_i32 : i32 loc(#loc9)
    %8 = arith.remsi %0, %4 : i32 loc(#loc10)
    %9 = arith.remsi %8, %7 : i32 loc(#loc11)
    %10 = arith.addi %5, %9 : i32 loc(#loc12)
    %11 = arith.divsi %8, %7 : i32 loc(#loc13)
    %12 = arith.muli %10, %c128_i32 : i32 loc(#loc14)
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc15)
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> loc(#loc15)
    %15 = tt.splat %12 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc16)
    %16 = tt.splat %12 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> loc(#loc16)
    %17 = arith.addi %15, %13 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc16)
    %18 = arith.addi %16, %14 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> loc(#loc16)
    %19 = tt.splat %arg3 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc17)
    %20 = arith.remsi %17, %19 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc17)
    %21 = arith.muli %11, %c256_i32 : i32 loc(#loc18)
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> loc(#loc19)
    %23 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc19)
    %24 = tt.splat %21 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> loc(#loc20)
    %25 = tt.splat %21 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc20)
    %26 = arith.addi %24, %22 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> loc(#loc20)
    %27 = arith.addi %25, %23 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc20)
    %28 = tt.splat %arg4 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc21)
    %29 = arith.remsi %27, %28 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc21)
    %30 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc22)
    %31 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked> loc(#loc23)
    %32 = arith.muli %30, %31 : tensor<128x1xi32, #blocked> loc(#loc23)
    %33 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc24)
    %34 = tt.expand_dims %33 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked> loc(#loc24)
    %35 = tt.broadcast %32 : tensor<128x1xi32, #blocked> -> tensor<128x16xi32, #blocked> loc(#loc25)
    %36 = tt.broadcast %34 : tensor<1x16xi32, #blocked> -> tensor<128x16xi32, #blocked> loc(#loc25)
    %37 = arith.addi %35, %36 : tensor<128x16xi32, #blocked> loc(#loc25)
    %38 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #blocked> loc(#loc26)
    %39 = tt.addptr %38, %37 : tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<128x16xi32, #blocked> loc(#loc26)
    %40 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc27)
    %41 = tt.expand_dims %40 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi32, #blocked1> loc(#loc27)
    %42 = tt.splat %arg7 : i32 -> tensor<16x1xi32, #blocked1> loc(#loc28)
    %43 = arith.muli %41, %42 : tensor<16x1xi32, #blocked1> loc(#loc28)
    %44 = tt.expand_dims %29 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1> loc(#loc29)
    %45 = tt.broadcast %43 : tensor<16x1xi32, #blocked1> -> tensor<16x256xi32, #blocked1> loc(#loc30)
    %46 = tt.broadcast %44 : tensor<1x256xi32, #blocked1> -> tensor<16x256xi32, #blocked1> loc(#loc30)
    %47 = arith.addi %45, %46 : tensor<16x256xi32, #blocked1> loc(#loc30)
    %48 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x256x!tt.ptr<f16>, #blocked1> loc(#loc31)
    %49 = tt.addptr %48, %47 : tensor<16x256x!tt.ptr<f16>, #blocked1>, tensor<16x256xi32, #blocked1> loc(#loc31)
    %50 = arith.addi %arg5, %c15_i32 : i32 loc(#loc59)
    %51 = arith.divsi %50, %c16_i32 : i32 loc(#loc60)
    %52 = arith.muli %arg7, %c16_i32 : i32 loc(#loc33)
    %53 = tt.splat %52 : i32 -> tensor<16x256xi32, #blocked1> loc(#loc34)
    %54 = tt.splat %arg5 : i32 -> tensor<1x16xi32, #blocked> loc(#loc35)
    %55 = arith.cmpi slt, %34, %54 : tensor<1x16xi32, #blocked> loc(#loc35)
    %56 = tt.broadcast %55 : tensor<1x16xi1, #blocked> -> tensor<128x16xi1, #blocked> loc(#loc36)
    %57 = tt.load %39, %56, %cst_0 : tensor<128x16x!tt.ptr<f16>, #blocked> loc(#loc36)
    %58 = triton_gpu.local_alloc %57 {allocation.offset = 0 : i32} : (tensor<128x16xf16, #blocked>) -> !tt.memdesc<128x16xf16, #shared, #triton_gpu.shared_memory> loc(#loc36)
    %59 = tt.splat %arg5 : i32 -> tensor<16x1xi32, #blocked1> loc(#loc37)
    %60 = arith.cmpi slt, %41, %59 : tensor<16x1xi32, #blocked1> loc(#loc37)
    %61 = tt.broadcast %60 : tensor<16x1xi1, #blocked1> -> tensor<16x256xi1, #blocked1> loc(#loc38)
    %62 = tt.load %49, %61, %cst_1 : tensor<16x256x!tt.ptr<f16>, #blocked1> loc(#loc38)
    %63 = triton_gpu.local_alloc %62 {allocation.offset = 4096 : i32} : (tensor<16x256xf16, #blocked1>) -> !tt.memdesc<16x256xf16, #shared1, #triton_gpu.shared_memory> loc(#loc38)
    %64 = tt.addptr %39, %cst : tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<128x16xi32, #blocked> loc(#loc39)
    %65 = tt.addptr %49, %53 : tensor<16x256x!tt.ptr<f16>, #blocked1>, tensor<16x256xi32, #blocked1> loc(#loc34)
    %66 = arith.subi %51, %c1_i32 : i32 loc(#loc40)
    cf.br ^bb1(%c0_i32, %cst_2, %64, %65 : i32, tensor<128x256xf32, #mma>, tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<16x256x!tt.ptr<f16>, #blocked1>) loc(#loc40)
  ^bb1(%67: i32 loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":143:22), %68: tensor<128x256xf32, #mma> loc(unknown), %69: tensor<128x16x!tt.ptr<f16>, #blocked> loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":151:18), %70: tensor<16x256x!tt.ptr<f16>, #blocked1> loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":152:18)):  // 2 preds: ^bb0, ^bb2
    %71 = arith.cmpi slt, %67, %66 : i32 loc(#loc40)
    cf.cond_br %71, ^bb2, ^bb3 loc(#loc40)
  ^bb2:  // pred: ^bb1
    %72 = arith.addi %67, %c1_i32 : i32 loc(#loc40)
    %73 = arith.muli %72, %c16_i32 : i32 loc(#loc41)
    %74 = arith.subi %arg5, %73 : i32 loc(#loc42)
    %75 = tt.splat %74 : i32 -> tensor<1x16xi32, #blocked> loc(#loc35)
    %76 = arith.cmpi slt, %34, %75 : tensor<1x16xi32, #blocked> loc(#loc35)
    %77 = tt.broadcast %76 : tensor<1x16xi1, #blocked> -> tensor<128x16xi1, #blocked> loc(#loc36)
    %78 = tt.load %69, %77, %cst_0 : tensor<128x16x!tt.ptr<f16>, #blocked> loc(#loc36)
    %79 = tt.splat %74 : i32 -> tensor<16x1xi32, #blocked1> loc(#loc37)
    %80 = arith.cmpi slt, %41, %79 : tensor<16x1xi32, #blocked1> loc(#loc37)
    %81 = tt.broadcast %80 : tensor<16x1xi1, #blocked1> -> tensor<16x256xi1, #blocked1> loc(#loc38)
    %82 = tt.load %70, %81, %cst_1 : tensor<16x256x!tt.ptr<f16>, #blocked1> loc(#loc38)
    %83 = triton_gpu.local_load %58 : !tt.memdesc<128x16xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc36)
    %84 = triton_gpu.local_load %63 : !tt.memdesc<16x256xf16, #shared1, #triton_gpu.shared_memory> -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc38)
    %85 = tt.dot %83, %84, %68 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xf32, #mma> loc(#loc43)
    %86 = tt.addptr %69, %cst : tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<128x16xi32, #blocked> loc(#loc39)
    %87 = tt.addptr %70, %53 : tensor<16x256x!tt.ptr<f16>, #blocked1>, tensor<16x256xi32, #blocked1> loc(#loc34)
    triton_gpu.local_store %78, %58 : tensor<128x16xf16, #blocked> -> !tt.memdesc<128x16xf16, #shared, #triton_gpu.shared_memory> loc(#loc36)
    triton_gpu.local_store %82, %63 : tensor<16x256xf16, #blocked1> -> !tt.memdesc<16x256xf16, #shared1, #triton_gpu.shared_memory> loc(#loc38)
    %88 = arith.addi %67, %c1_i32 : i32 loc(#loc40)
    cf.br ^bb1(%88, %85, %86, %87 : i32, tensor<128x256xf32, #mma>, tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<16x256x!tt.ptr<f16>, #blocked1>) loc(#loc40)
  ^bb3:  // pred: ^bb1
    %89 = triton_gpu.local_load %58 : !tt.memdesc<128x16xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc36)
    %90 = triton_gpu.local_load %63 : !tt.memdesc<16x256xf16, #shared1, #triton_gpu.shared_memory> -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc38)
    %91 = tt.dot %89, %90, %68 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xf32, #mma> loc(#loc43)
    %92 = arith.truncf %91 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma> loc(#loc44)
    %93 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma> loc(#loc45)
    %94 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #mma> loc(#loc46)
    %95 = arith.muli %94, %93 : tensor<128x1xi32, #mma> loc(#loc46)
    %96 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #mma> loc(#loc47)
    %97 = tt.addptr %96, %95 : tensor<128x1x!tt.ptr<f16>, #mma>, tensor<128x1xi32, #mma> loc(#loc47)
    %98 = tt.expand_dims %26 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi32, #mma> loc(#loc48)
    %99 = tt.broadcast %97 : tensor<128x1x!tt.ptr<f16>, #mma> -> tensor<128x256x!tt.ptr<f16>, #mma> loc(#loc49)
    %100 = tt.broadcast %98 : tensor<1x256xi32, #mma> -> tensor<128x256xi32, #mma> loc(#loc49)
    %101 = tt.addptr %99, %100 : tensor<128x256x!tt.ptr<f16>, #mma>, tensor<128x256xi32, #mma> loc(#loc49)
    %102 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #mma> loc(#loc50)
    %103 = arith.cmpi slt, %93, %102 : tensor<128x1xi32, #mma> loc(#loc50)
    %104 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #mma> loc(#loc51)
    %105 = arith.cmpi slt, %98, %104 : tensor<1x256xi32, #mma> loc(#loc51)
    %106 = tt.broadcast %103 : tensor<128x1xi1, #mma> -> tensor<128x256xi1, #mma> loc(#loc52)
    %107 = tt.broadcast %105 : tensor<1x256xi1, #mma> -> tensor<128x256xi1, #mma> loc(#loc52)
    %108 = arith.andi %106, %107 : tensor<128x256xi1, #mma> loc(#loc52)
    tt.store %101, %92, %108 : tensor<128x256x!tt.ptr<f16>, #mma> loc(#loc53)
    tt.return loc(#loc54)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":114:24)
#loc3 = loc("/opt/conda/envs/py_3.10/lib/python3.10/site-packages/triton/language/standard.py":40:22)
#loc4 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":115:27)
#loc5 = loc("/opt/conda/envs/py_3.10/lib/python3.10/site-packages/triton/language/standard.py":40:28)
#loc6 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":116:27)
#loc7 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":118:22)
#loc8 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":120:35)
#loc9 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":120:48)
#loc10 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":121:34)
#loc11 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":121:54)
#loc12 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":121:27)
#loc13 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":122:40)
#loc14 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":131:23)
#loc15 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":131:51)
#loc16 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":131:38)
#loc17 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":131:68)
#loc18 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":132:23)
#loc19 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":132:51)
#loc20 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":132:38)
#loc21 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":132:68)
#loc22 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":134:30)
#loc23 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":134:41)
#loc24 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":134:60)
#loc25 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":134:53)
#loc26 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":134:22)
#loc27 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":135:29)
#loc28 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":135:40)
#loc29 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":135:60)
#loc30 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":135:52)
#loc31 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":135:22)
#loc32 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":143:33)
#loc33 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":152:33)
#loc35 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":146:51)
#loc36 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":146:20)
#loc37 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":147:51)
#loc38 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":147:20)
#loc41 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":146:59)
#loc42 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":146:55)
#loc43 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":149:35)
#loc44 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":157:23)
#loc45 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":163:41)
#loc46 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":163:33)
#loc47 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":163:21)
#loc48 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":163:72)
#loc49 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":163:52)
#loc50 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":164:33)
#loc51 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":164:58)
#loc52 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":164:39)
#loc53 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":165:21)
#loc54 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_gemm.py":165:4)
#loc55 = loc(callsite(#loc3 at #loc4))
#loc56 = loc(callsite(#loc5 at #loc4))
#loc57 = loc(callsite(#loc3 at #loc6))
#loc58 = loc(callsite(#loc5 at #loc6))
#loc59 = loc(callsite(#loc3 at #loc32))
#loc60 = loc(callsite(#loc5 at #loc32))
"""

def get_ttgir_kernel():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(gemm_ttgir)
        f.flush()
        kernel = triton.compile(f.name)
        return kernel
    

def matmul_ttgir(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c




# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

def test_correctness():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    # Bigger tolerance for AMD MI200 devices.
    # MI200 devices use reduced precision fp16 and bf16 and flush input and
    # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    rtol = 1e-2 if is_hip_mi200() else 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol)


if __name__ == "__main__":
    test_correctness()
