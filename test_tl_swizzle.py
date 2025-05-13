import numpy as np
import torch
import triton
import triton.language as tl
import re
import pytest


@triton.jit
def swizzle_kernel(output, col_num):
    num_tiles = tl.num_programs(0)
    pid = tl.program_id(0)
    pid_i = pid // col_num
    pid_j = pid % col_num 
    out_i, out_j = tl.swizzle2d(pid_i, pid_j, col_num, col_num, 4)
    # tl.device_print("index = %d, %d, %d, %d", pid_i, pid_j, out_i, out_j)
    tl.store(output + pid, out_i)
    tl.store(output + num_tiles + pid, out_j)


def test_swizzle(size, col_num):
    num_tiles = size
    output = torch.zeros((num_tiles * 2), dtype = torch.int32, device = 'cuda') - 1
    swizzle_kernel[(num_tiles,)](output, col_num)

    # row_num = 8
    # col_num = 8
    out0 = output[0:num_tiles]
    out1 = output[num_tiles:]
    # for i in range(row_num):
    #     print(f"{out0[i * col_num : (i + 1) * col_num]}")
    #     print(f"{out1[i * col_num : (i + 1) * col_num]}")
    #     print()

    shuffled = torch.zeros((num_tiles, ), dtype = torch.int32, device = 'cuda') - 1

    for i in range(num_tiles):
        in_j = i % col_num
        in_i = i // col_num

        out_i = out0[i]
        out_j = out1[i]
        shuffled[out_i * col_num + out_j] = i

    for i in range(num_tiles // col_num):
        print(f"{shuffled[i * col_num : (i + 1) * col_num]}")

test_swizzle(80, 10)
