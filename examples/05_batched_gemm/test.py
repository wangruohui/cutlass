import time

import torch
from torch.utils.cpp_extension import load

cpp = load(
    name="gemm",
    sources=["batched_gemm_torch.cu"],
    extra_include_paths=[
        "/home/wangruohui.p/cutlass/examples/common/",
        "/home/wangruohui.p/cutlass/include",
    ],
    verbose=True,
)


dtype = torch.float
device = torch.device(5)

a = torch.rand((5,2,3), dtype=dtype, device=device)
b = torch.rand((5,3,4), dtype=dtype, device=device)
c0 = torch.zeros((5,2,4), dtype=dtype, device=device)
c1 = torch.zeros((5,2,4), dtype=dtype, device=device)
c2 = torch.zeros((5,2,4), dtype=dtype, device=device)

print(a.size())
print(b.size())

torch.bmm(a, b, out=c0)
cpp.grouped_gemm_batch(a, b, c1)
# cpp.grouped_gemm_array(a, b, c2)

print(c0)
print(c1)
# print(c2)

assert torch.allclose(c0, c1)
# assert torch.allclose(c0, c2)

# torch.cuda.synchronize()
# start = time.monotonic()
# torch.bmm(a, b, out=c0)
# print(c0)
# end = time.monotonic()
# print(f"torch.mm: {end - start}")

# # torch.cuda.synchronize()
# # start = time.monotonic()
# # cpp.grouped_gemm_batch(a, b, c1)
# # print(c1)
# # end = time.monotonic()
# # print(f"batch: {end - start}")

# torch.cuda.synchronize()
# start = time.monotonic()
# cpp.grouped_gemm_array(a, b, c2)
# print(c2)
# end = time.monotonic()
# print(f"array: {end - start}")
