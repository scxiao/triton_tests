import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
import torch


@triton.jit
def mxgemm_kernel(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_scale: tl.constexpr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        fptype :tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    DIV_FACTOR: tl.constexpr = 2 if fptype == 4 else 1
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K // DIV_FACTOR)
    offs_scale_k = tl.arange(0, BLOCK_K // 32)
    a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    k = 0
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        valid_k = offs_k < k_remaining
        a = tl.load(a_ptrs)#, mask=valid_k[None, :], other=0.)
        b = tl.load(b_ptrs)#, mask=valid_k[:, None], other=0.)
        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)
        if fptype == 4:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator)
        elif fptype == 85: # e5m2
            accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e5m2", accumulator)
        else: # e4m3fn
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b, scale_b, "e4m3", accumulator)

        a_ptrs += (BLOCK_K // DIV_FACTOR) * stride_ak
        b_ptrs += (BLOCK_K // DIV_FACTOR) * stride_bk
        a_scale_ptr += BLOCK_K // 32
        b_scale_ptr += BLOCK_K // 32


    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


def fp8e8m0_to_float32(scale):
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23
    scale = scale.view(torch.float32)
    return scale


def torch_gemm_mxfp(a, b, a_scale, b_scale, dtype_src_str):
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)

    a_scale_f32 = a_scale_f32.repeat_interleave(32, dim=1)
    b_scale_f32 = b_scale_f32.repeat_interleave(32, dim=1)
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)

    # b_scales are always col major
    b_scale_f32 = b_scale_f32.T.contiguous()

    a = a_f32 * a_scale_f32
    b = b_f32 * b_scale_f32

    ref_out = torch.matmul(a, b).to(torch.float32)

    return ref_out


# Default to float16. Change this to bfloat16 to use bf16 datatypes
def get_ptype(dtype):
    ptype = "fp16"
    if dtype == "bfloat16":
        ptype = "bf16"
    elif dtype == "float8_e4m3fn":
        ptype = "fp8e4nv"
    elif dtype == "float8_e5m2":
        ptype = "fp8e5"
    elif dtype == "float4":
        ptype = "u8"

    return ptype


def generate_configs():
    base_configs = [
        {"M": 1024, "N": 1024, "K": 1024, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":128, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE": 'float4'},
        {"M": 1024, "N": 1024, "K": 1024, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":128, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE": 'float8_e4m3fn'},
        {"M": 1024, "N": 1024, "K": 1024, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":128, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE": 'float8_e5m2'},
    ]
    return base_configs


def triton_gemm_mxfp(config):
    M = config['M']
    N = config['N']
    K = config['K']
    BLOCK_M = config['BLOCK_M']
    BLOCK_N = config['BLOCK_N']
    BLOCK_K = config['BLOCK_K']
    # NUM_CTAS = config['NUM_CTAS']
    num_warps = config['NUM_WARPS']
    dtype = config['DTYPE']
    num_stages = 2


    fp_type_flag = 4 
    if dtype=="float8_e4m3fn":
        fp_type_flag = 84
    elif dtype == "float8_e5m2":
        fp_type_flag = 85

    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)
    if (dtype != "float4"):
        torch_type = getattr(torch, dtype)
        a = (torch.randint(1, 10, (M, K))).to(torch_type)
        b = (torch.randint(1, 10, (K, N))).to(torch_type)
    else:
        a = MXFP4Tensor(data = torch.randint(1,5, (M,K)))
        b = MXFP4Tensor(data = torch.randint(1,5, (K,N)))

    a_scale = torch.randint(127,130, (M, K // 32), dtype=torch.uint8)
    b_scale = torch.randint(127,130, (N, K // 32), dtype=torch.uint8)

    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, dtype)
    c = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    # mxfp4 input needs packed along the k dim, i.e., two mxfp4 are packed in one uint8
    if dtype == 'float4':
        a = a.to_packed_tensor(dim=1)
        b = b.to_packed_tensor(dim=0)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    a = a.cuda()
    b = b.cuda()
    a_scale = a_scale.cuda()
    b_scale = b_scale.cuda()
    mxgemm_kernel[grid](  #
        a, b, c,  #
        a_scale, b_scale,  #
        M, N, K,  #
        K // 32, #stride_scale: tl.constexpr
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        fp_type_flag,
        BLOCK_M, BLOCK_N, BLOCK_K,  #
        num_stages=num_stages,
        num_warps=num_warps)

    torch.testing.assert_close(c.cpu(), c_ref)

if __name__ == '__main__':
    triton_gemm_mxfp(generate_configs()[2])