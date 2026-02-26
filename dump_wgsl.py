"""Dump the generated WGSL for the GEMM kernel to see what gets emitted."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def linear_loop_fp16w_kernel(X, W, Bias, Y, K, stride_x, stride_w, N,
                              BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    num_chunks = (K + BLOCK_K - 1) // BLOCK_K

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK_K
        ks = off + tl.arange(0, BLOCK_K)
        mask = ks < K
        x = tl.load(X + row * stride_x + ks, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + col * stride_w + ks, mask=mask, other=0.0).to(tl.float32)
        acc += x * w
    dot = tl.sum(acc, axis=0)
    b = tl.load(Bias + col).to(tl.float32)
    tl.store(Y + row * N + col, dot + b)


from triton.backends.webgpu.compiler import WebGPUBackend

target = GPUTarget("webgpu", 0, 32)
backend = WebGPUBackend(target)

src = triton.compiler.ASTSource(
    fn=linear_loop_fp16w_kernel,
    signature={
        "X": "*fp32",
        "W": "*fp16",
        "Bias": "*fp32",
        "Y": "*fp32",
        "K": "i32",
        "stride_x": "i32",
        "stride_w": "i32",
        "N": "i32",
    },
    constexprs={"BLOCK_K": 64},
)

options = backend.parse_options(dict())
compiled = triton.compile(src, target=target)

# Get the WGSL source
wgsl = compiled.asm.get("wgsl", "")
if not wgsl:
    print("Available asm keys:", list(compiled.asm.keys()))
    for k, v in compiled.asm.items():
        if isinstance(v, str) and len(v) < 5000:
            print(f"\n=== {k} ===")
            print(v[:2000])
else:
    print("=== Generated WGSL (GEMM kernel, fp32 acts + fp16 weights) ===")
    print(wgsl)
    print(f"\n=== Total lines: {len(wgsl.splitlines())} ===")
