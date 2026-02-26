"""
WebGPU backend performance benchmarks.

Measures kernel execution time and effective throughput for common GPU
workload patterns.  Each test prints timing info; no hard thresholds are
asserted (results are hardware-dependent), but correctness is always
verified.

Benchmarks:
    1. Vector add      — memory bandwidth (read-read-write)
    2. Vector copy     — raw copy bandwidth (read-write)
    3. Vector scale    — compute-light bandwidth (read-write)
    4. Reduction sum   — cross-thread reduction
    5. Softmax         — row-wise softmax (exp, reduce, elementwise)
    6. Atomic add      — contended atomic accumulation
    7. Chained ops     — multi-operation fusion (add-mul-relu)
    8. Kernel launch   — launch overhead / latency

Run:
    pytest python/test/unit/language/test_webgpu_perf.py -v -s
"""
import time
import numpy as np
import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton.backends.webgpu.llvm_to_wgsl import translate_llvm_to_wgsl
from triton.backends.webgpu.dawn_runner import DawnRunner, HAS_DAWN

pytestmark = pytest.mark.skipif(not HAS_DAWN(), reason="Dawn WebGPU library not available")

WEBGPU_TARGET = GPUTarget("webgpu", 0, 32)

# ============================================================================
# Infrastructure
# ============================================================================

_runner = None


def get_runner():
    global _runner
    if _runner is None:
        _runner = DawnRunner()
    return _runner


def compile_webgpu(fn, signature, constexprs=None):
    if constexprs is None:
        constexprs = {}
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=WEBGPU_TARGET)


def compile_and_translate(fn, signature, constexprs, num_warps=4, warp_size=32):
    """Compile & translate, return (result, runner)."""
    sig_no_ce = {k: v for k, v in signature.items() if v != 'constexpr'}
    k = compile_webgpu(fn, signature, constexprs)
    runner = get_runner()
    result = translate_llvm_to_wgsl(k.asm['llir'], sig_no_ce, num_warps, warp_size,
                                     use_native_subgroups=runner.has_subgroups)
    return result, runner


def run_kernel(result, runner, grid, buffers, scalars=None):
    """Execute a pre-compiled kernel."""
    return runner.run_kernel(
        wgsl_code=result.wgsl,
        buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields,
        workgroup_size=result.workgroup_size,
        grid=grid,
        buffers=buffers,
        scalars=scalars or {},
    )


def assert_close(actual, expected, rtol=1e-5, atol=1e-5, name="output"):
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol,
        err_msg=f"{name}: max_diff={np.max(np.abs(actual - expected)):.6e}"
    )


def _bench(fn, warmup=2, iters=10):
    """Run *fn* and return per-iteration time in milliseconds."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000, out


def _bw_str(bytes_transferred, ms):
    return f"{bytes_transferred / (ms / 1000) / 1e9:.2f} GB/s"


# ============================================================================
# 1. Vector Add — memory bandwidth (2 reads + 1 write)
# ============================================================================

@pytest.mark.parametrize("N", [1 << 16, 1 << 18, 1 << 20])
def test_vector_add_bandwidth(N):
    BLOCK_SIZE = 256
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    @triton.jit
    def kernel(X, Y, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(X + offs, mask=mask)
        b = tl.load(Y + offs, mask=mask)
        tl.store(Z + offs, a + b, mask=mask)

    sig = {'X': '*fp32', 'Y': '*fp32', 'Z': '*fp32',
           'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    result, runner = compile_and_translate(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE})
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    def go():
        return run_kernel(result, runner, grid,
                          {'X': x, 'Y': y, 'Z': out}, {'n': N})

    ms, outputs = _bench(go)
    bw = 3 * N * 4  # 2 reads + 1 write
    print(f"\n  VecAdd  N={N:>10,}: {ms:7.2f} ms  {_bw_str(bw, ms)}")
    assert_close(outputs['Z'][:N], x + y, name="vec_add")


# ============================================================================
# 2. Vector Copy — raw copy bandwidth (1 read + 1 write)
# ============================================================================

@pytest.mark.parametrize("N", [1 << 16, 1 << 18, 1 << 20])
def test_vector_copy_bandwidth(N):
    BLOCK_SIZE = 256
    x = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(X + offs, mask=mask)
        tl.store(Z + offs, a, mask=mask)

    sig = {'X': '*fp32', 'Z': '*fp32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    result, runner = compile_and_translate(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE})
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    def go():
        return run_kernel(result, runner, grid, {'X': x, 'Z': out}, {'n': N})

    ms, outputs = _bench(go)
    bw = 2 * N * 4
    print(f"\n  Copy    N={N:>10,}: {ms:7.2f} ms  {_bw_str(bw, ms)}")
    assert_close(outputs['Z'][:N], x, name="copy")


# ============================================================================
# 3. Vector Scale — compute-light bandwidth (1 read + 1 write + 1 mul)
# ============================================================================

@pytest.mark.parametrize("N", [1 << 16, 1 << 18, 1 << 20])
def test_vector_scale_bandwidth(N):
    BLOCK_SIZE = 256
    x = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    SCALE = np.float32(3.14)

    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(X + offs, mask=mask)
        tl.store(Z + offs, a * 3.14, mask=mask)

    sig = {'X': '*fp32', 'Z': '*fp32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    result, runner = compile_and_translate(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE})
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    def go():
        return run_kernel(result, runner, grid, {'X': x, 'Z': out}, {'n': N})

    ms, outputs = _bench(go)
    bw = 2 * N * 4
    print(f"\n  Scale   N={N:>10,}: {ms:7.2f} ms  {_bw_str(bw, ms)}")
    assert_close(outputs['Z'][:N], x * SCALE, name="scale", rtol=1e-4)


# ============================================================================
# 4. Reduction Sum — single-workgroup tree reduction
# ============================================================================

@pytest.mark.parametrize("N", [32, 64, 128, 256])
def test_reduction_sum(N):
    BLOCK_SIZE = N
    x = np.random.randn(N).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)

    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(X + offs, mask=mask)
        s = tl.sum(a, axis=0)
        # Only thread 0 writes the result
        tl.store(Z, s)

    # WebGPU max workgroup size is 256 → cap num_warps at 8
    num_warps = min(8, max(1, BLOCK_SIZE // 32))
    sig = {'X': '*fp32', 'Z': '*fp32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    result, runner = compile_and_translate(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE},
                                           num_warps=num_warps)

    def go():
        return run_kernel(result, runner, (1,), {'X': x, 'Z': out}, {'n': N})

    ms, outputs = _bench(go)
    expected = np.array([x.sum()], dtype=np.float32)
    print(f"\n  ReduceSum N={N:>6,}: {ms:7.2f} ms")
    assert_close(outputs['Z'][:1], expected, rtol=1e-3, atol=1e-3, name="reduce_sum")


# ============================================================================
# 5. Softmax — row-wise (load, exp, reduce, normalize)
# ============================================================================

@pytest.mark.parametrize("N_COLS", [64, 128, 256])
def test_softmax(N_COLS):
    N_ROWS = 64
    BLOCK_SIZE = N_COLS
    x = np.random.randn(N_ROWS, N_COLS).astype(np.float32)
    out = np.zeros_like(x)

    @triton.jit
    def kernel(X, Z, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        row_ptr = X + row * n_cols + offs
        a = tl.load(row_ptr, mask=mask, other=-float('inf'))
        a_max = tl.max(a, axis=0)
        a_exp = tl.exp(a - a_max)
        a_sum = tl.sum(a_exp, axis=0)
        out = a_exp / a_sum
        tl.store(Z + row * n_cols + offs, out, mask=mask)

    # WebGPU max workgroup size is 256 → cap num_warps at 8
    num_warps = min(8, max(1, BLOCK_SIZE // 32))
    sig = {'X': '*fp32', 'Z': '*fp32', 'n_cols': 'i32', 'BLOCK_SIZE': 'constexpr'}
    result, runner = compile_and_translate(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE},
                                           num_warps=num_warps)
    grid = (N_ROWS,)

    def go():
        return run_kernel(result, runner, grid,
                          {'X': x.ravel(), 'Z': out.ravel()}, {'n_cols': N_COLS})

    ms, outputs = _bench(go)
    # NumPy reference
    x_max = x.max(axis=1, keepdims=True)
    e = np.exp(x - x_max)
    expected = e / e.sum(axis=1, keepdims=True)
    actual = outputs['Z'][:N_ROWS * N_COLS].reshape(N_ROWS, N_COLS)
    print(f"\n  Softmax {N_ROWS}x{N_COLS}: {ms:7.2f} ms")
    assert_close(actual, expected, rtol=1e-4, atol=1e-4, name="softmax")


# ============================================================================
# 6. Atomic Add — contended accumulation
# ============================================================================

@pytest.mark.parametrize("num_wg", [4, 16, 64])
def test_atomic_add_throughput(num_wg):
    N = 64
    BLOCK_SIZE = 64
    NUM_WARPS = 2
    x = np.ones(N, dtype=np.int32)
    out = np.zeros(N, dtype=np.int32)

    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        v = tl.load(X + offs, mask=mask)
        tl.atomic_add(Z + offs, v, mask=mask)

    sig = {'X': '*i32', 'Z': '*i32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    result, runner = compile_and_translate(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE},
                                           num_warps=NUM_WARPS)

    def go():
        return run_kernel(result, runner, (num_wg,),
                          {'X': x, 'Z': np.zeros(N, dtype=np.int32)}, {'n': N})

    ms, outputs = _bench(go)
    expected = np.full(N, num_wg, dtype=np.int32)
    ops = num_wg * N
    print(f"\n  AtomicAdd WG={num_wg:>3}: {ms:7.2f} ms  ({ops/ms/1e3:.1f} M atomic-ops/s)")
    np.testing.assert_array_equal(outputs['Z'][:N], expected)


# ============================================================================
# 7. Chained Ops — add + mul + relu fusion
# ============================================================================

@pytest.mark.parametrize("N", [1 << 16, 1 << 18, 1 << 20])
def test_chained_ops(N):
    BLOCK_SIZE = 256
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    @triton.jit
    def kernel(X, Y, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(X + offs, mask=mask)
        b = tl.load(Y + offs, mask=mask)
        c = (a + b) * a
        # ReLU
        c = tl.where(c > 0, c, 0.0)
        tl.store(Z + offs, c, mask=mask)

    sig = {'X': '*fp32', 'Y': '*fp32', 'Z': '*fp32',
           'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    result, runner = compile_and_translate(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE})
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    def go():
        return run_kernel(result, runner, grid,
                          {'X': x, 'Y': y, 'Z': out}, {'n': N})

    ms, outputs = _bench(go)
    ref = (x + y) * x
    ref = np.maximum(ref, 0)
    bw = 3 * N * 4
    print(f"\n  Chained N={N:>10,}: {ms:7.2f} ms  {_bw_str(bw, ms)}")
    assert_close(outputs['Z'][:N], ref, name="chained", rtol=1e-4)


# ============================================================================
# 8. Kernel Launch Latency — tiny kernel, measure dispatch overhead
# ============================================================================

def test_kernel_launch_latency():
    N = 32
    BLOCK_SIZE = 32
    x = np.ones(N, dtype=np.float32)
    out = np.zeros(N, dtype=np.float32)

    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(X + offs, mask=mask)
        tl.store(Z + offs, a, mask=mask)

    sig = {'X': '*fp32', 'Z': '*fp32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    result, runner = compile_and_translate(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE},
                                           num_warps=1)

    def go():
        return run_kernel(result, runner, (1,), {'X': x, 'Z': out}, {'n': N})

    ms, outputs = _bench(go, warmup=5, iters=50)
    print(f"\n  Launch latency: {ms:.3f} ms  ({ms * 1000:.0f} us)")
    assert_close(outputs['Z'][:N], x, name="latency")


# ============================================================================
# 9. Scaling — same kernel, increasing problem size
# ============================================================================

def test_bandwidth_scaling():
    """Measure how bandwidth scales with N from 2^12 to 2^20."""
    BLOCK_SIZE = 256
    results = []

    @triton.jit
    def kernel(X, Y, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(X + offs, mask=mask)
        b = tl.load(Y + offs, mask=mask)
        tl.store(Z + offs, a + b, mask=mask)

    sig = {'X': '*fp32', 'Y': '*fp32', 'Z': '*fp32',
           'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    result, runner = compile_and_translate(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE})

    print(f"\n  {'N':>12}  {'Time (ms)':>10}  {'BW (GB/s)':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}")

    for exp in range(12, 21, 2):
        N = 1 << exp
        x = np.random.randn(N).astype(np.float32)
        y = np.random.randn(N).astype(np.float32)
        out = np.zeros(N, dtype=np.float32)
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        def go(x=x, y=y, out=out, grid=grid):
            return run_kernel(result, runner, grid,
                              {'X': x, 'Y': y, 'Z': out}, {'n': N})

        ms, outputs = _bench(go, warmup=2, iters=5)
        bw_bytes = 3 * N * 4
        bw_gbs = bw_bytes / (ms / 1000) / 1e9
        results.append((N, ms, bw_gbs))
        assert_close(outputs['Z'][:N], x + y, name=f"scale_{N}")
        print(f"  {N:>12,}  {ms:>10.2f}  {bw_gbs:>10.2f}")

    # Sanity: bandwidth should generally increase with N
    # (small sizes are latency-bound, large sizes approach peak BW)
    assert len(results) > 0


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--tb=short'])
