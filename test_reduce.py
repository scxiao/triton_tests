import pytest
import torch
import triton
import triton.language as tl

@triton.jit
def add_constant_kernel(M, val, out_ptr):
   # acc:tl.float32 = 0.
    acc = 0.0
    for i in tl.range(0, M):
        acc += val

    # store final
    # tl.store(out_ptr, acc.to(out_ptr.type.element_ty))
    tl.store(out_ptr, acc)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [10, 200, 4096, 8192])
def test_accum_constant(M, out_dtype):

    device = 'cuda'
    out = torch.zeros((1,), dtype=out_dtype, device=device)
    val = 3.1415
    grid = (1,)
    add_constant_kernel[grid](
        M,
        val,
        out,
    )

    triton_val = out

    exact_val = 0.0
    for _ in range(M):
        exact_val += val
    # cast to float32 for a fair direct comparison
    ref_val = torch.tensor([exact_val], dtype=torch.float32, device='cuda')

    if out_dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-3, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    err = ref_val - triton_val

    val_tensor = torch.tensor([val], dtype=torch.float32, device='cuda')
    print(f"val(3.1415) = {val_tensor[0]}")
    err_tensor = torch.tensor([err], dtype=torch.float32, device='cuda')
    print(f"err = {err_tensor[0]}")
    print(f"M={M}, Triton val={triton_val[0]}, Ref val={ref_val[0]}, abs error={err}, dtype = {out_dtype}")



    val_tensor = torch.tensor([val], dtype=torch.float32, device='cuda')
    print(f"val(3.1415) = {bin(val_tensor.view(torch.uint32)[0])}")
    err_tensor = torch.tensor([err], dtype=torch.float32, device='cuda')
    print(f"err = {bin(err_tensor.view(torch.uint32)[0])}")
    print(f"M={M}, Triton val={bin(triton_val.view(torch.uint32)[0])}, Ref val={bin(ref_val.view(torch.uint32)[0])}, abs error={err}, dtype = {out_dtype}")
    assert err < atol, f"Naive sum error too large: {err}, output dytpe = {out_dtype}, atol = {atol}"
