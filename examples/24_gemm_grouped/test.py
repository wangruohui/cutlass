import torch

from torch.utils.cpp_extension import load
import torch.utils.benchmark as benchmark


from torch.profiler import profile, record_function, ProfilerActivity

import time

cpp = load(
    name="grouped_gemm",
    sources=["gemm_grouped_torch.cu"],
    extra_include_paths=[
        "/home/wangruohui.p/cutlass/examples/common/",
        "/home/wangruohui.p/cutlass/include",
        "/home/wangruohui.p/cutlass/tools/util/include",
    ],
    verbose=True,
    # extra_cuda_cflags=[
    #     "-forward-unknown-to-host-compiler",
    #     "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    # ]
    extra_cuda_cflags=[
        "-forward-unknown-to-host-compiler",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-O3",
        "-DNDEBUG",
    ],
)


dtype = torch.half
device = torch.device(0)

As = []
Bs = []
Cs = []
Ds = []

for i in range(1, 100, 10):
    align = 8
    size = (i + 1) * align
    a = torch.rand(align, size, dtype=dtype, device=device)
    b = torch.rand(size, align, dtype=dtype, device=device)
    c = torch.zeros(align, align, dtype=dtype, device=device)
    d = torch.zeros(align, align, dtype=dtype, device=device)
    As.append(a)
    Bs.append(b)
    Cs.append(c)
    Ds.append(d)
    print(f"{i//10} {align}x{size}x{align}")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.monotonic()
        cpp.grouped_gemm(As, Bs, Cs, True)
        torch.cuda.synchronize()
        end = time.monotonic()
        print(f"cutlass: {(end - start)*1000} (ms)")

        torch.cuda.synchronize()
        start = time.monotonic()
        for i, (a, b, d) in enumerate(zip(As, Bs, Ds)):
            torch.mm(a, b, out=d)
        torch.cuda.synchronize()
        end = time.monotonic()
        print(f"naive  : {(end - start)*1000} (ms)")

prof.export_chrome_trace("grouped gemm.json")

torch.set_printoptions(linewidth=200)

from utils import diff_tensor

Cs = torch.stack(Cs)
Ds = torch.stack(Ds)

if not torch.allclose(Cs, Ds, atol=1e-3, rtol=1e-3):
    diff_tensor(Cs, Ds)
