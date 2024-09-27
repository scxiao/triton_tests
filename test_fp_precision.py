# This test is from Alex Kranias
import torch
import triton
import triton.language as tl
import pytest
import pdb

@triton.jit
def many_ops_triton(x_ptr,
                    y_ptr,
                    o_ptr,
                    M: tl.constexpr,
                    K: tl.constexpr,
                    N: tl.constexpr,
                    mult: tl.constexpr,
                    IMITATE_PYTORCH: tl.constexpr,
                    DTYPE: tl.constexpr,
                    DO_MULTIPLY: tl.constexpr,
                    DO_SIGMOID: tl.constexpr,
                    DO_COS: tl.constexpr,
                    DO_EXPONENT: tl.constexpr,
                    DO_SQRT: tl.constexpr
                ):
    """
    x_ptr: pointer to an (M, K) tensor [input]
    y_ptr: pointer to an (K, N) tensor [input]

    o_ptr: pointer to an (M, N) tensor [output]

    M: int matrix shape
    K: int matrix shape
    N: int matrix shape

    mult: multiplication factor for multiplication operation

    IMITATE_PYTORCH: {
        0: no casting after ops, 
        1: cast to original dtype after every op
    }
    DTYPE: {
        0: fp16, 
        1: fp32, 
        2: fp64
    }
    """
    # Set input dtype (we will cast back to this for the output)
    input_dtype = tl.float16 if DTYPE==0 else tl.float32 if DTYPE==1 else None

    x_block_range = tl.arange(0, M)[:, None]*K + tl.arange(0, K)[None, :]
    y_block_range = tl.arange(0, K)[:, None]*N + tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + x_block_range)
    y = tl.load(y_ptr + y_block_range)

    # Multiply
    if DO_MULTIPLY:
        x = x * mult
        y = y * mult
        if IMITATE_PYTORCH:
            x = x.to(input_dtype)
            y = y.to(input_dtype)

    # Sigmoid
    if DO_SIGMOID:
        x = tl.sigmoid(x + 0.0) # +0.0 cause tl.sigmoid requires a fp32 and 0.0 is fp32 by default so if dtype if fp16 will become fp32
        y = tl.sigmoid(y + 0.0)
        if IMITATE_PYTORCH:
            x = x.to(input_dtype)
            y = y.to(input_dtype)

    # Cos
    if DO_COS:
        x = tl.cos(x + 0.0)     # +0.0 because requires fp32 or fp64
        y = tl.cos(y + 0.0)
        if IMITATE_PYTORCH:
            x = x.to(input_dtype)
            y = y.to(input_dtype)

    # Exponentiate
    if DO_EXPONENT:
        log2_e = 1.4426950408889634  # log2(e)
        x = tl.exp2(log2_e * x)
        y = tl.exp2(log2_e * y)
        if IMITATE_PYTORCH:
            x = x.to(input_dtype)
            y = y.to(input_dtype)

    # Sqrt
    if DO_SQRT:
        x = tl.sqrt(x + 0.0)    # +0.0 because requires fp32 or fp64
        y = tl.sqrt(y + 0.0)
        if IMITATE_PYTORCH:
            x = x.to(input_dtype)
            y = y.to(input_dtype)

    # Matmul
    o_block_range = tl.arange(0, M)[:, None]*N + tl.arange(0, N)[None, :]
    o = tl.dot(x, y) # tl.dot always outputs input dtype. ALSO REQUIRES INPUT SHAPES M >= 16, N >= 16 and K >= 16
    if IMITATE_PYTORCH:
        x = x.to(input_dtype)
        y = y.to(input_dtype)

    # o = tl.dot(x, y, out_dtype=input_dtype) # FUSE CAST INTO DOT

    tl.store(o_ptr + o_block_range, o)

def many_ops_torch(x: torch.Tensor,
                   y: torch.Tensor,
                   out: torch.Tensor,
                   M: int,
                   K: int,
                   N: int,
                   mult: float,
                   DO_MULTIPLY: bool,
                   DO_SIGMOID: bool,
                   DO_COS: bool,
                   DO_EXPONENT: bool,
                   DO_SQRT: bool
                ):
    
    # Multiply
    if DO_MULTIPLY:
        x = x * mult
        y = y * mult

    # Sigmoid
    if DO_SIGMOID:
        x = torch.sigmoid(x)
        y = torch.sigmoid(y)

    # Cos
    if DO_COS:
        x = torch.cos(x)
        y = torch.cos(y)

    # Exponentiate
    if DO_EXPONENT:
        x = torch.exp(x)
        y = torch.exp(y)

    # Sqrt
    if DO_SQRT:
        x = torch.sqrt(x)
        y = torch.sqrt(y)

    # Matmul
    out[:] = torch.matmul(x, y) # stores in place

@pytest.mark.parametrize("seed", [i for i in range(1)])  # seed for rand num generator
@pytest.mark.parametrize("M", [16, 32])
@pytest.mark.parametrize("K", [16, 32, 64]) # 64 seems to cause some issues
@pytest.mark.parametrize("N", [16, 32])
@pytest.mark.parametrize("mult", [0.001, 1.5251]) # mult = [0, 2.99]
@pytest.mark.parametrize("IMITATE_PYTORCH", [1]) # 0 = no casting (not imitating pytorch), 1 = cast after every op (imitating pytorch)
@pytest.mark.parametrize("DTYPE", [0]) # 0 = fp16, 1 = fp32
@pytest.mark.parametrize("DO_MULTIPLY", [0, 1])  # Include multiplication
@pytest.mark.parametrize("DO_SIGMOID", [0, 1])  # Include sigmoid
@pytest.mark.parametrize("DO_COS", [0, 1])  # Include cosine
@pytest.mark.parametrize("DO_EXPONENT", [0, 1])  # Include exponentiation
@pytest.mark.parametrize("DO_SQRT", [0, 1])  # Include square root
def test_many_ops(seed, M, K, N, mult, IMITATE_PYTORCH, DTYPE, DO_MULTIPLY, DO_SIGMOID, DO_COS, DO_EXPONENT, DO_SQRT):
    """
    Test reproducability of PyTorch results with a Triton kernel implementing various math operations.

    Each operation can be individually enabled or disabled using the respective parameters. The test will compare
    the results from Triton and PyTorch to ensure they match within a specified tolerance.

    Args:
        seed (int): Random seed for reproducibility.
        M (int): Number of rows for the first input tensor.
        K (int): Number of columns for the first input tensor and rows for the second.
        N (int): Number of columns for the second input tensor.
        mult (float): Multiplication factor for the input tensors.
        IMITATE_PYTORCH (int): If 1, cast tensors back to their original dtype after each operation, if 0 does not cast until very end.
        DTYPE (int): Data type of the input tensors (0 for fp16, 1 for fp32).
        DO_MULTIPLY (int): If 1, include multiplication in the operations, if 0 does not.
        DO_SIGMOID (int): If 1, include sigmoid activation in the operations, if 0 does not.
        DO_COS (int): If 1, include cosine transformation in the operations, if 0 does not.
        DO_EXPONENT (int): If 1, include exponentiation in the operations, if 0 does not.
        DO_SQRT (int): If 1, include square root in the operations, if 0 does not.
    """

    # Misc parameters
    torch.set_printoptions(precision=6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)

    input_dtype = torch.float16 if DTYPE==0 else torch.float32 if DTYPE==1 else None

    x = torch.rand(M, K, dtype=input_dtype, device=device)
    y = torch.rand(K, N, dtype=input_dtype, device=device)

    grid = (1,)
    out = torch.zeros(M, N, dtype=input_dtype, device=device)
    out_torch = torch.zeros(M, N, dtype=input_dtype, device=device)

    with torch.cuda.device(x.device):
        many_ops_triton[grid](x, y, out, M, K, N, mult, IMITATE_PYTORCH, DTYPE, DO_MULTIPLY, DO_SIGMOID, DO_COS, DO_EXPONENT, DO_SQRT)
        many_ops_torch(x, y, out_torch, M, K, N, mult, DO_MULTIPLY, DO_SIGMOID, DO_COS, DO_EXPONENT, DO_SQRT)

    print("torch", out_torch)
    print("out", out)

    print("torch - out", (out_torch-out))

    assert torch.allclose(out_torch, out, atol=0) # tensors must match exactly

