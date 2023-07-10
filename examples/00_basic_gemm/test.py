import torch

from torch.utils.cpp_extension import load


import time

cpp = load(
    name="gemm",
    sources=["basic_gemm_torch.cu"],
    extra_include_paths=[
        "/home/wangruohui.p/cutlass/examples/common/",
        "/home/wangruohui.p/cutlass/include",
    ],
    # verbose=True,
)


dtype = torch.half
device = torch.device(0)

a = torch.tensor([[2, 3, 4], [1, 2, 2]], dtype=dtype, device=device)
b = torch.tensor([[1, 4, 1, 3], [2, 3, 5, 6], [1, 3, 2, 4]], dtype=dtype, device=device)
c0 = torch.zeros(2, 4, dtype=dtype, device=device)
c2 = torch.zeros(2, 4, dtype=dtype, device=device)

print(a.size())
print(b.size())

# gemm1(a, b, c1)

torch.cuda.synchronize()
start = time.monotonic()
torch.mm(a, b, out=c0)
print(c0)
end = time.monotonic()
print(f"torch.mm: {end - start}")

torch.cuda.synchronize()
start = time.monotonic()
cpp.gemm2(a, b, c2)
print(c2)
end = time.monotonic()
print(f"cutlass: {end - start}")
# print(a @ b)
# print(a.T @ b.T)
