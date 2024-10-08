import torch

import triton
import triton.language as tl
import tempfile

vecadd_ir = """
#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#loc = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":8:0)
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 0 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":8:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":8:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":8:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":8:0)) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c1024_i32 : i32 loc(#loc3)
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked> loc(#loc4)
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked> loc(#loc5)
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked> loc(#loc5)
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked> loc(#loc6)
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked> loc(#loc6)
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc7)
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked> loc(#loc7)
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc8)
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc9)
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked> loc(#loc9)
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc10)
    %13 = arith.addf %9, %12 : tensor<1024xf32, #blocked> loc(#loc11)
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc12)
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked> loc(#loc12)
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc13)
    tt.return loc(#loc14)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":17:24)
#loc3 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":22:24)
#loc4 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":23:41)
#loc5 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":23:28)
#loc6 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":25:21)
#loc7 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":28:24)
#loc8 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":28:16)
#loc9 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":29:24)
#loc10 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":29:16)
#loc11 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":30:17)
#loc12 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":32:26)
#loc13 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":32:35)
#loc14 = loc("/workspace/projects/triton-openai/python/test/test_ttgir_vecadd.py":32:4)
"""

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def get_kernel():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(vecadd_ir)
        f.flush()
        vecadd_kernel = triton.compile(f.name)
        return vecadd_kernel
    

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = triton.cdiv(n_elements, 1024)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    # add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    vecadd_kernel = get_kernel()
    vecadd_kernel[(grid,1,1)](x, y, output, n_elements)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

def main():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')

if __name__ == "__main__":
    main()
