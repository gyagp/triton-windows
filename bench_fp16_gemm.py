"""Benchmark: Triton-compiled fp16 GEMM vs hand-crafted WGSL fp16 GEMM.

Compares:
  1. linear_loop_fp16w_kernel (Triton compiled) — current production kernel
  2. WGSL_FP16_GEMM_KERNEL (hand-crafted) — subgroupAdd + vec4 dot + unpack2x16float

Uses FLUX Klein dimensions: K=3072, N∈{3072, 9216}
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "python", "examples", "webgpu"))
os.environ.setdefault("TRITON_WEBGPU_DAWN_PATH",
                      os.path.join(os.path.dirname(__file__),
                                   "python", "examples", "webgpu",
                                   "libs", "dawn.dll"))

import numpy as np
from triton.backends.webgpu.dawn_runner import DawnRunner, GPUBuffer

runner = DawnRunner()

# -------------------------------------------------------------------------
# Compile Triton kernel
# -------------------------------------------------------------------------
from triton.backends.compiler import GPUTarget
from common.model_base import KernelCache
from common.kernels import linear_loop_fp16w_kernel

cache = KernelCache()

LB = 128  # LOOP_BLOCK

sig = {
    'X': '*fp32', 'W': '*fp16', 'Bias': '*fp32', 'Y': '*fp32',
    'K': 'i32', 'stride_x': 'i32', 'stride_w': 'i32',
    'N': 'i32', 'BLOCK_K': 'constexpr',
}

triton_result = cache.get_or_compile(
    linear_loop_fp16w_kernel, sig,
    {'BLOCK_K': LB}, num_warps=max(1, LB // 32))

# -------------------------------------------------------------------------
# Load WGSL kernel
# -------------------------------------------------------------------------
from common.wgsl_kernels import (
    WGSL_FP16_GEMM_KERNEL, FP16_GEMM_BINDINGS,
    pack_fp16_gemm_params, FP16_GEMM_TILE_N,
)


def bench_triton(x_gpu, w_gpu, bias_gpu, T, N, K, warmup=5, iters=20):
    """Benchmark Triton-compiled fp16 GEMM."""
    out_buf = np.zeros(T * N, dtype=np.float32)

    for _ in range(warmup):
        cache.run(triton_result, grid=(T, N),
                  buffers={'X': x_gpu, 'W': w_gpu, 'Bias': bias_gpu,
                           'Y': out_buf},
                  scalars={'K': K, 'stride_x': K, 'stride_w': K, 'N': N})

    t0 = time.perf_counter()
    for _ in range(iters):
        cache.run(triton_result, grid=(T, N),
                  buffers={'X': x_gpu, 'W': w_gpu, 'Bias': bias_gpu,
                           'Y': out_buf},
                  scalars={'K': K, 'stride_x': K, 'stride_w': K, 'N': N})
    dt = (time.perf_counter() - t0) / iters
    return dt, cache.run(triton_result, grid=(T, N),
                         buffers={'X': x_gpu, 'W': w_gpu, 'Bias': bias_gpu,
                                  'Y': out_buf},
                         scalars={'K': K, 'stride_x': K, 'stride_w': K, 'N': N})['Y'].reshape(T, N)


def bench_wgsl(x_gpu, w_u32_gpu, bias_gpu, T, N, K, warmup=5, iters=20):
    """Benchmark hand-crafted WGSL fp16 GEMM."""
    params = pack_fp16_gemm_params(K, N)
    params_gpu = runner.upload_to_gpu(params, f"fp16gemm_bench_params_{K}_{N}")

    out_buf = np.zeros(T * N, dtype=np.float32)

    for _ in range(warmup):
        runner.run_kernel(
            wgsl_code=WGSL_FP16_GEMM_KERNEL,
            buffer_bindings=FP16_GEMM_BINDINGS,
            param_fields=[],
            workgroup_size=256,
            grid=(T, (N + FP16_GEMM_TILE_N - 1) // FP16_GEMM_TILE_N),
            buffers={'X': x_gpu, 'W': w_u32_gpu, 'Bias': bias_gpu,
                     'Y': out_buf, '_params_': params_gpu},
            scalars={})

    t0 = time.perf_counter()
    for _ in range(iters):
        runner.run_kernel(
            wgsl_code=WGSL_FP16_GEMM_KERNEL,
            buffer_bindings=FP16_GEMM_BINDINGS,
            param_fields=[],
            workgroup_size=256,
            grid=(T, (N + FP16_GEMM_TILE_N - 1) // FP16_GEMM_TILE_N),
            buffers={'X': x_gpu, 'W': w_u32_gpu, 'Bias': bias_gpu,
                     'Y': out_buf, '_params_': params_gpu},
            scalars={})
    dt = (time.perf_counter() - t0) / iters
    result = runner.run_kernel(
        wgsl_code=WGSL_FP16_GEMM_KERNEL,
        buffer_bindings=FP16_GEMM_BINDINGS,
        param_fields=[],
        workgroup_size=256,
        grid=(T, (N + FP16_GEMM_TILE_N - 1) // FP16_GEMM_TILE_N),
        buffers={'X': x_gpu, 'W': w_u32_gpu, 'Bias': bias_gpu,
                 'Y': out_buf, '_params_': params_gpu},
        scalars={})
    return dt, result['Y'].reshape(T, N)


# -------------------------------------------------------------------------
# Run benchmark
# -------------------------------------------------------------------------
def run_test(T, N, K):
    print(f"\n{'='*60}")
    print(f"T={T}, N={N}, K={K}  (weight size: {N*K*2/1e6:.1f} MB fp16)")
    print(f"{'='*60}")

    np.random.seed(42)
    x = np.random.randn(T, K).astype(np.float32) * 0.01
    w = np.random.randn(N, K).astype(np.float16)
    bias = np.zeros(N, dtype=np.float32)

    # Upload for Triton kernel (fp16 buffer)
    x_gpu = runner.upload_to_gpu(x.ravel(), f"X_{T}_{K}")
    w_gpu = runner.upload_to_gpu(w.ravel(), f"W_{N}_{K}")
    bias_gpu = runner.upload_to_gpu(bias, f"bias_{N}")

    # Upload for WGSL kernel (same fp16 data viewed as u32)
    w_u32 = w.ravel().view(np.uint32)
    w_u32_gpu = runner.upload_to_gpu(w_u32, f"W_u32_{N}_{K}")

    # Benchmark
    dt_triton, y_triton = bench_triton(x_gpu, w_gpu, bias_gpu, T, N, K)
    dt_wgsl, y_wgsl = bench_wgsl(x_gpu, w_u32_gpu, bias_gpu, T, N, K)

    # Correctness check
    max_err = np.max(np.abs(y_triton - y_wgsl))
    rel_err = max_err / (np.max(np.abs(y_triton)) + 1e-8)

    # Reference (numpy)
    y_ref = x @ w.astype(np.float32).T + bias
    triton_vs_ref = np.max(np.abs(y_triton - y_ref))
    wgsl_vs_ref = np.max(np.abs(y_wgsl - y_ref))

    print(f"  Triton compiled: {dt_triton*1000:.3f} ms")
    print(f"  WGSL hand-craft: {dt_wgsl*1000:.3f} ms")
    speedup = dt_triton / dt_wgsl if dt_wgsl > 0 else float('inf')
    print(f"  Speedup (WGSL/Triton): {speedup:.2f}x")
    print(f"  Max error (WGSL vs Triton): {max_err:.6f} (rel: {rel_err:.6f})")
    print(f"  Triton vs NumPy ref: {triton_vs_ref:.6f}")
    print(f"  WGSL vs NumPy ref:   {wgsl_vs_ref:.6f}")


# FLUX Klein dimensions
run_test(T=1, N=3072, K=3072)   # Output projection
run_test(T=1, N=9216, K=3072)   # QKV / FF projection
run_test(T=1, N=3072, K=9216)   # FF down projection
run_test(T=64, N=3072, K=3072)  # T=64 batch (512×512 img)
run_test(T=64, N=9216, K=3072)  # T=64 batch + wide
