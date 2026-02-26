"""
WebGPU backend core tests — mirrors test_core.py for the CUDA backend.

Tests compile Triton kernels for the WebGPU target, translate LLVM IR to
WGSL, and execute compute shaders on the GPU via Dawn (Google's native
WebGPU implementation), verifying results against NumPy references.

WebGPU/WGSL type support:
    - f32 (float32) — primary float type
    - i32 (int32)   — primary integer type

Run:
    pytest python/test/unit/language/test_webgpu_core.py -v
"""
import inspect
import textwrap
import time

import numpy as np
import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton.backends.webgpu.llvm_to_wgsl import translate_llvm_to_wgsl
from triton.backends.webgpu.dawn_runner import DawnRunner, HAS_DAWN

# Skip entire module if Dawn is not available
pytestmark = pytest.mark.skipif(not HAS_DAWN(), reason="Dawn WebGPU library not available")

WEBGPU_TARGET = GPUTarget("webgpu", 0, 32)

# ============================================================================
# Type constants for parametrization
# ============================================================================

webgpu_float_dtypes = ['float32']
webgpu_int_dtypes = ['int32']
webgpu_dtypes = webgpu_float_dtypes + webgpu_int_dtypes

DTYPE_TO_SIG = {'float32': 'fp32', 'int32': 'i32'}
DTYPE_TO_NP = {'float32': np.float32, 'int32': np.int32}

# ============================================================================
# Test infrastructure
# ============================================================================

_runner = None


def get_runner():
    global _runner
    if _runner is None:
        _runner = DawnRunner()
    return _runner


def compile_webgpu(fn, signature, constexprs=None):
    """Compile a Triton kernel for WebGPU."""
    if constexprs is None:
        constexprs = {}
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=WEBGPU_TARGET)


def compile_and_run(fn, signature, constexprs, grid, buffers, scalars=None,
                    num_warps=4, warp_size=32):
    """Compile → translate → execute on GPU, return output buffers."""
    scalars = scalars or {}
    sig_no_constexpr = {k: v for k, v in signature.items() if v != 'constexpr'}
    k = compile_webgpu(fn, signature, constexprs)
    llir = k.asm['llir']
    runner = get_runner()
    result = translate_llvm_to_wgsl(llir, sig_no_constexpr, num_warps, warp_size,
                                     use_native_subgroups=runner.has_subgroups)
    return runner.run_kernel(
        wgsl_code=result.wgsl,
        buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields,
        workgroup_size=result.workgroup_size,
        grid=grid,
        buffers=buffers,
        scalars=scalars,
    )


def assert_close(actual, expected, rtol=1e-5, atol=1e-5, name="output"):
    """Assert numpy arrays are close with a descriptive message."""
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol,
        err_msg=f"{name}: max_diff={np.max(np.abs(actual - expected)):.6e}"
    )


def patch_kernel(template, to_replace):
    """Replace strings in a kernel template — matches test_core.py helper."""
    kernel = triton.JITFunction(template.fn)
    src = kernel.src
    for key, value in to_replace.items():
        src = src.replace(key, value)
    kernel._unsafe_update_src(src)
    return kernel


# ============================================================================
# Helpers for running simple element-wise kernels
# ============================================================================

BLOCK = 256


def _run_binary(op_expr, dtype, n=1024, numpy_expr=None, x_range=None, y_range=None,
                filter_y=None, out_dtype=None):
    """Run a binary element-wise kernel: z = <op_expr> where x, y are inputs."""
    np_dtype = DTYPE_TO_NP[dtype]
    sig_type = DTYPE_TO_SIG[dtype]
    out_dtype = out_dtype or dtype
    out_np = DTYPE_TO_NP[out_dtype]
    out_sig = DTYPE_TO_SIG[out_dtype]

    if dtype == 'int32':
        lo, hi = (x_range or (-100, 100))
        x = np.random.randint(lo, hi, n).astype(np_dtype)
        lo, hi = (y_range or (-100, 100))
        y = np.random.randint(lo, hi, n).astype(np_dtype)
    else:
        x = np.random.randn(n).astype(np_dtype)
        y = np.random.randn(n).astype(np_dtype)

    if filter_y is not None:
        y[filter_y(y)] = 1

    z = np.zeros(n, dtype=out_np)
    z_ref = eval(numpy_expr or op_expr)

    @triton.jit
    def kernel(X, Y, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        y = tl.load(Y + offs, mask=mask)
        z = GENERATE_TEST_HERE
        tl.store(Z + offs, z, mask=mask)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': op_expr})
    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': f'*{sig_type}', 'Y': f'*{sig_type}',
                   'Z': f'*{out_sig}', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Y': y, 'Z': z},
        scalars={'n_elements': n},
    )
    return results['Z'][:n], z_ref


def _run_unary(expr, dtype, n=1024, numpy_expr=None, x_range=None, out_dtype=None):
    """Run a unary element-wise kernel: z = <expr> where x is input."""
    np_dtype = DTYPE_TO_NP[dtype]
    sig_type = DTYPE_TO_SIG[dtype]
    out_dtype = out_dtype or dtype
    out_np = DTYPE_TO_NP[out_dtype]
    out_sig = DTYPE_TO_SIG[out_dtype]

    if dtype == 'int32':
        lo, hi = (x_range or (-100, 100))
        x = np.random.randint(lo, hi, n).astype(np_dtype)
    else:
        x = np.random.randn(n).astype(np_dtype)

    z = np.zeros(n, dtype=out_np)
    z_ref = eval(numpy_expr or expr)

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        z = GENERATE_TEST_HERE
        tl.store(Z + offs, z, mask=mask)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': f'*{sig_type}', 'Z': f'*{out_sig}',
                   'n_elements': 'i32', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    return results['Z'][:n], z_ref


# ============================================================================
# Binary Arithmetic Ops  (cf. test_bin_op in test_core.py)
# ============================================================================

@pytest.mark.parametrize("op", ['+', '-', '*', '/'])
def test_bin_op_float(op):
    """Float32 binary arithmetic: x <op> y."""
    filter_y = (lambda y: y == 0) if op == '/' else None
    actual, expected = _run_binary(f'x {op} y', 'float32', filter_y=filter_y)
    assert_close(actual, expected, rtol=1e-4, atol=1e-5, name=f"float_{op}")


@pytest.mark.parametrize("op", ['+', '-', '*'])
def test_bin_op_int(op):
    """Int32 binary arithmetic: x <op> y."""
    actual, expected = _run_binary(f'x {op} y', 'int32')
    np.testing.assert_array_equal(actual, expected, err_msg=f"int_{op}")


def test_floordiv_int():
    """Int32 floor division (Python //) — Triton uses truncation division."""
    actual, expected = _run_binary(
        'x // y', 'int32',
        numpy_expr='np.trunc(x / y).astype(np.int32)',
        filter_y=lambda y: y == 0,
    )
    np.testing.assert_array_equal(actual, expected, err_msg="int_floordiv")


def test_modulo_int():
    """Int32 modulo (Python %)."""
    actual, expected = _run_binary(
        'x % y', 'int32',
        numpy_expr='np.fmod(x, y).astype(np.int32)',
        filter_y=lambda y: y == 0,
    )
    np.testing.assert_array_equal(actual, expected, err_msg="int_mod")


# ============================================================================
# Bitwise Ops  (cf. test_bitwise_op in test_core.py)
# ============================================================================

@pytest.mark.parametrize("op", ['&', '|', '^'])
def test_bitwise_op(op):
    """Int32 bitwise operations."""
    actual, expected = _run_binary(f'x {op} y', 'int32')
    np.testing.assert_array_equal(actual, expected, err_msg=f"bitwise_{op}")


@pytest.mark.parametrize("op", ['<<', '>>'])
def test_shift_op(op):
    """Int32 shift operations."""
    actual, expected = _run_binary(
        f'x {op} y', 'int32',
        x_range=(1, 1000), y_range=(0, 8),
    )
    np.testing.assert_array_equal(actual, expected, err_msg=f"shift_{op}")


def test_bitwise_not():
    """Int32 bitwise NOT (~x)."""
    actual, expected = _run_unary('~x', 'int32')
    np.testing.assert_array_equal(actual, expected, err_msg="bitwise_not")


# ============================================================================
# Compare Ops  (cf. test_compare_op in test_core.py)
# ============================================================================

@pytest.mark.parametrize("dtype", webgpu_dtypes)
@pytest.mark.parametrize("op", ['==', '!=', '>', '<', '>=', '<='])
def test_compare_op(dtype, op):
    """Comparison operators for float32 and int32."""
    np_dtype = DTYPE_TO_NP[dtype]
    sig_type = DTYPE_TO_SIG[dtype]
    n = 1024

    if dtype == 'int32':
        x = np.random.randint(-100, 100, n).astype(np_dtype)
        y = np.random.randint(-100, 100, n).astype(np_dtype)
    else:
        x = np.random.randn(n).astype(np_dtype)
        y = np.random.randn(n).astype(np_dtype)
    # Add some equal values to test == and !=
    y[:10] = x[:10]

    z_ref = eval(f'(x {op} y).astype(np.int32)')
    z = np.zeros(n, dtype=np.int32)

    @triton.jit
    def kernel(X, Y, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        y = tl.load(Y + offs, mask=mask)
        z = GENERATE_TEST_HERE
        tl.store(Z + offs, z, mask=mask)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'x {op} y'})
    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': f'*{sig_type}', 'Y': f'*{sig_type}',
                   'Z': '*i32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Y': y, 'Z': z},
        scalars={'n_elements': n},
    )
    actual = results['Z'][:n]
    # Boolean results: Triton returns -1 for true, 0 for false (i32 comparison)
    # or 1 for true, 0 for false — normalize for comparison
    actual_bool = actual != 0
    expected_bool = z_ref != 0
    np.testing.assert_array_equal(actual_bool, expected_bool,
                                  err_msg=f"compare {dtype} {op}")


@pytest.mark.parametrize("op", ['==', '!=', '>', '<', '>=', '<='])
def test_compare_op_nan(op):
    """Float32 comparison with NaN values."""
    n = 256
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    x[:n // 3] = float('nan')
    y[n // 3:2 * n // 3] = float('nan')

    z_ref = eval(f'(x {op} y).astype(np.int32)')
    z = np.zeros(n, dtype=np.int32)

    @triton.jit
    def kernel(X, Y, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        y = tl.load(Y + offs, mask=mask)
        z = GENERATE_TEST_HERE
        tl.store(Z + offs, z, mask=mask)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'x {op} y'})
    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Y': '*fp32', 'Z': '*i32',
                   'n_elements': 'i32', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Y': y, 'Z': z},
        scalars={'n_elements': n},
    )
    actual_bool = results['Z'][:n] != 0
    expected_bool = z_ref != 0
    np.testing.assert_array_equal(actual_bool, expected_bool,
                                  err_msg=f"compare_nan {op}")


# ============================================================================
# Unary Ops  (cf. test_unary_op in test_core.py)
# ============================================================================

def test_neg_float():
    """Float32 negation: -x."""
    actual, expected = _run_unary('-x', 'float32')
    assert_close(actual, expected, atol=0, name="neg_float")


def test_neg_int():
    """Int32 negation: -x."""
    actual, expected = _run_unary('-x', 'int32')
    np.testing.assert_array_equal(actual, expected, err_msg="neg_int")


def test_abs_float():
    """Float32 absolute value."""
    actual, expected = _run_unary('tl.abs(x)', 'float32', numpy_expr='np.abs(x)')
    assert_close(actual, expected, atol=0, name="abs_float")


# ============================================================================
# Math Ops  (cf. test_math_op in test_core.py)
# ============================================================================

@pytest.mark.parametrize("expr, np_expr", [
    ('tl.math.exp(x)', 'np.exp(x)'),
    ('tl.math.log(x)', 'np.log(np.abs(x) + 1e-6)'),
    ('tl.math.sqrt(x)', 'np.sqrt(np.abs(x))'),
])
def test_math_op(expr, np_expr):
    """Float32 math functions."""
    n = 1024
    x = np.abs(np.random.randn(n).astype(np.float32)) + 0.01
    z = np.zeros(n, dtype=np.float32)
    z_ref = eval(np_expr)

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        z = GENERATE_TEST_HERE
        tl.store(Z + offs, z, mask=mask)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, rtol=1e-4, atol=1e-5, name=expr)


@pytest.mark.parametrize("expr, np_expr", [
    ('tl.math.sin(x)', 'np.sin(x)'),
    ('tl.math.cos(x)', 'np.cos(x)'),
])
def test_trig_op(expr, np_expr):
    """Float32 trigonometric functions."""
    n = 1024
    x = (np.random.randn(n).astype(np.float32) * 3.0)
    z = np.zeros(n, dtype=np.float32)
    z_ref = eval(np_expr)

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        z = GENERATE_TEST_HERE
        tl.store(Z + offs, z, mask=mask)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, rtol=1e-4, atol=1e-5, name=expr)


# ============================================================================
# Where / Select  (cf. test_where in test_core.py)
# ============================================================================

@pytest.mark.parametrize("dtype", webgpu_dtypes)
def test_where(dtype):
    """tl.where(cond, a, b) for float32 and int32."""
    np_dtype = DTYPE_TO_NP[dtype]
    sig_type = DTYPE_TO_SIG[dtype]
    n = 1024

    if dtype == 'int32':
        x = np.random.randint(-100, 100, n).astype(np_dtype)
        y = np.random.randint(-100, 100, n).astype(np_dtype)
    else:
        x = np.random.randn(n).astype(np_dtype)
        y = np.random.randn(n).astype(np_dtype)

    cond = np.random.randint(0, 2, n).astype(np.int32)
    z_ref = np.where(cond, x, y)
    z = np.zeros(n, dtype=np_dtype)

    @triton.jit
    def kernel(Cond, X, Y, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        c = tl.load(Cond + offs, mask=mask)
        x = tl.load(X + offs, mask=mask)
        y = tl.load(Y + offs, mask=mask)
        z = tl.where(c, x, y)
        tl.store(Z + offs, z, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'Cond': '*i32', 'X': f'*{sig_type}', 'Y': f'*{sig_type}',
                   'Z': f'*{sig_type}', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'Cond': cond, 'X': x, 'Y': y, 'Z': z},
        scalars={'n_elements': n},
    )
    if dtype == 'int32':
        np.testing.assert_array_equal(results['Z'][:n], z_ref, err_msg="where_int")
    else:
        assert_close(results['Z'][:n], z_ref, atol=0, name="where_float")


# ============================================================================
# Broadcast  (cf. test_broadcast in test_core.py)
# ============================================================================

def test_broadcast_scalar():
    """Broadcast scalar to vector: scalar + vector."""
    n = 1024
    scalar_val = 3.14
    x = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    z_ref = x + scalar_val

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        z = x + 3.14
        tl.store(Z + offs, z, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, rtol=1e-5, name="broadcast_scalar")


def test_broadcast_row():
    """Broadcast 1D row across 2D: add row vector to each row of a matrix.

    Uses a 2D-in-1D pattern since WebGPU buffers are flat.
    """
    N = 128  # Must match default workgroup size (num_warps=4 * warp_size=32)
    M = 32
    x = np.random.randn(M * N).astype(np.float32)
    row = np.random.randn(N).astype(np.float32)
    z = np.zeros(M * N, dtype=np.float32)

    # Expected: x reshaped (M, N) + row[None, :], flattened
    z_ref = (x.reshape(M, N) + row[None, :]).flatten()

    @triton.jit
    def kernel(X, Row, Z, M: tl.constexpr, N: tl.constexpr):
        pid = tl.program_id(0)  # row index
        row_offs = tl.arange(0, N)
        x_ptr = X + pid * N + row_offs
        r_ptr = Row + row_offs
        x_val = tl.load(x_ptr)
        r_val = tl.load(r_ptr)
        tl.store(Z + pid * N + row_offs, x_val + r_val)

    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Row': '*fp32', 'Z': '*fp32',
                   'M': 'constexpr', 'N': 'constexpr'},
        constexprs={'M': M, 'N': N},
        grid=(M,),
        buffers={'X': x, 'Row': row, 'Z': z},
    )
    assert_close(results['Z'][:M * N], z_ref, rtol=1e-5, name="broadcast_row")


# ============================================================================
# Full / Arange / Zeros  (cf. test_full, test_arange in test_core.py)
# ============================================================================

@pytest.mark.parametrize("dtype", webgpu_dtypes)
def test_full(dtype):
    """tl.full — create tensor filled with a constant."""
    n = 128
    fill_val = 42 if dtype == 'int32' else 3.14
    np_dtype = DTYPE_TO_NP[dtype]
    sig_type = DTYPE_TO_SIG[dtype]
    z = np.zeros(n, dtype=np_dtype)
    z_ref = np.full(n, fill_val, dtype=np_dtype)

    @triton.jit
    def kernel(Z, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        z = GENERATE_TEST_HERE
        tl.store(Z + offs, z)

    tl_dtype = 'tl.int32' if dtype == 'int32' else 'tl.float32'
    kernel = patch_kernel(kernel,
                          {'GENERATE_TEST_HERE': f'tl.full([BLOCK_SIZE], {fill_val}, {tl_dtype})'})
    results = compile_and_run(
        fn=kernel,
        signature={'Z': f'*{sig_type}', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': n},
        grid=(1,),
        buffers={'Z': z},
    )
    if dtype == 'int32':
        np.testing.assert_array_equal(results['Z'][:n], z_ref, err_msg="full_int")
    else:
        assert_close(results['Z'][:n], z_ref, atol=1e-5, name="full_float")


@pytest.mark.parametrize("start", [0, 1, 7, 16])
def test_arange(start):
    """tl.arange — generate sequential integer range."""
    n = 128
    z = np.zeros(n, dtype=np.int32)
    z_ref = np.arange(start, start + n, dtype=np.int32)

    @triton.jit
    def kernel(Z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        val = tl.arange(START, END)
        tl.store(Z + offs, val)

    results = compile_and_run(
        fn=kernel,
        signature={'Z': '*i32', 'BLOCK': 'constexpr',
                   'START': 'constexpr', 'END': 'constexpr'},
        constexprs={'BLOCK': n, 'START': start, 'END': start + n},
        grid=(1,),
        buffers={'Z': z},
    )
    np.testing.assert_array_equal(results['Z'][:n], z_ref, err_msg=f"arange_{start}")


def test_zeros():
    """tl.zeros — create zero-filled tensor."""
    n = 128
    z = np.ones(n, dtype=np.float32)  # Pre-fill with 1s to verify zeros overwrites

    @triton.jit
    def kernel(Z, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        z = tl.zeros([BLOCK_SIZE], tl.float32)
        tl.store(Z + offs, z)

    results = compile_and_run(
        fn=kernel,
        signature={'Z': '*fp32', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': n},
        grid=(1,),
        buffers={'Z': z},
    )
    np.testing.assert_array_equal(results['Z'][:n], np.zeros(n, dtype=np.float32))


# ============================================================================
# Cast  (cf. test_cast in test_core.py)
# ============================================================================

def test_cast_float_to_int():
    """Cast float32 → int32 (truncation)."""
    n = 1024
    x = (np.random.randn(n) * 100).astype(np.float32)
    z = np.zeros(n, dtype=np.int32)
    z_ref = x.astype(np.int32)

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        z = x.to(tl.int32)
        tl.store(Z + offs, z, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*i32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    np.testing.assert_array_equal(results['Z'][:n], z_ref, err_msg="cast_f2i")


def test_cast_int_to_float():
    """Cast int32 → float32."""
    n = 1024
    x = np.random.randint(-1000, 1000, n).astype(np.int32)
    z = np.zeros(n, dtype=np.float32)
    z_ref = x.astype(np.float32)

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        z = x.to(tl.float32)
        tl.store(Z + offs, z, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*i32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, atol=0, name="cast_i2f")


# ============================================================================
# Reduce 1D  (cf. test_reduce1d in test_core.py)
# ============================================================================

@pytest.mark.parametrize("op", ['sum', 'max', 'min'])
@pytest.mark.parametrize("dtype", webgpu_dtypes)
@pytest.mark.parametrize("shape", [32, 64, 128, 256])
def test_reduce1d(op, dtype, shape):
    """1D reductions: sum, max, min for various block sizes."""
    np_dtype = DTYPE_TO_NP[dtype]
    sig_type = DTYPE_TO_SIG[dtype]

    if dtype == 'int32':
        x = np.random.randint(-100, 100, shape).astype(np_dtype)
    else:
        x = np.random.randn(shape).astype(np_dtype)

    numpy_op = {'sum': np.sum, 'max': np.max, 'min': np.min}[op]
    z_ref_scalar = numpy_op(x)

    z = np.zeros(1, dtype=np_dtype)

    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        GENERATE_TEST_HERE
        tl.store(Z, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'z = tl.{op}(x, axis=0)'})
    results = compile_and_run(
        fn=kernel,
        signature={'X': f'*{sig_type}', 'Z': f'*{sig_type}',
                   'BLOCK': 'constexpr'},
        constexprs={'BLOCK': shape},
        grid=(1,),
        buffers={'X': x, 'Z': z},
    )
    actual = results['Z'][0]
    if op == 'sum':
        assert_close(np.array([actual]), np.array([z_ref_scalar]),
                     rtol=0.01, atol=1e-3, name=f"reduce_{op}_{dtype}_{shape}")
    else:
        np.testing.assert_equal(actual, z_ref_scalar,
                                err_msg=f"reduce_{op}_{dtype}_{shape}")


# ============================================================================
# Masked Load / Store  (cf. test_masked_load in test_core.py)
# ============================================================================

@pytest.mark.parametrize("size_diff", [0, 1, 4, 15])
@pytest.mark.parametrize("other", [0.0, 1.0])
def test_masked_load(size_diff, other):
    """Load with mask and 'other' value — matches test_masked_load pattern."""
    block_size = 128
    input_size = block_size - size_diff
    x = np.random.randn(input_size).astype(np.float32)
    z = np.zeros(block_size, dtype=np.float32)

    # Reference: values from x, then padding with `other`
    z_ref = np.concatenate([x, np.full(size_diff, other, dtype=np.float32)])

    @triton.jit
    def kernel(X, Z, in_size, BLOCK: tl.constexpr, OTHER: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < in_size
        x = tl.load(X + offs, mask=mask, other=OTHER)
        tl.store(Z + offs, x)

    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'in_size': 'i32',
                   'BLOCK': 'constexpr', 'OTHER': 'constexpr'},
        constexprs={'BLOCK': block_size, 'OTHER': other},
        grid=(1,),
        buffers={'X': np.pad(x, (0, size_diff), constant_values=0), 'Z': z},
        scalars={'in_size': input_size},
    )
    assert_close(results['Z'][:block_size], z_ref, atol=0, name=f"masked_load_{size_diff}_{other}")


def test_masked_store():
    """Store with mask — only writes elements where mask is true."""
    n = 256
    BLOCK_SIZE = 256
    x = np.random.randn(n).astype(np.float32)
    z = np.full(n, -1.0, dtype=np.float32)

    # Only write first half
    half = n // 2
    z_ref = np.concatenate([x[:half], np.full(n - half, -1.0, dtype=np.float32)])

    @triton.jit
    def kernel(X, Z, half, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < half
        x = tl.load(X + offs)
        tl.store(Z + offs, x, mask=mask)

    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'half': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK_SIZE},
        grid=(1,),
        buffers={'X': x, 'Z': z},
        scalars={'half': half},
    )
    assert_close(results['Z'][:n], z_ref, atol=0, name="masked_store")


# ============================================================================
# Load / Store Patterns  (cf. test_load_store, test_strided_load/store)
# ============================================================================

def test_copy():
    """Simple memory copy (load + store)."""
    n = 2048
    x = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        val = tl.load(X + offs, mask=mask)
        tl.store(Z + offs, val, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], x, atol=0, name="copy")


def test_strided_load():
    """Strided access: read every other element."""
    n = 512
    x = np.random.randn(n * 2).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    z_ref = x[::2][:n]

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        src_offs = offs * 2
        src_mask = src_offs < (n_elements * 2)
        val = tl.load(X + src_offs, mask=src_mask)
        tl.store(Z + offs, val, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, atol=0, name="strided_load")


def test_indirect_load():
    """Indirect (gather) access via index array."""
    n = 256
    vals = np.random.randn(n).astype(np.float32)
    indices = np.random.permutation(n).astype(np.int32)
    z = np.zeros(n, dtype=np.float32)
    z_ref = vals[indices]

    @triton.jit
    def kernel(Vals, Idx, Z, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        idx = tl.load(Idx + offs)
        val = tl.load(Vals + idx)
        tl.store(Z + offs, val)

    results = compile_and_run(
        fn=kernel,
        signature={'Vals': '*fp32', 'Idx': '*i32', 'Z': '*fp32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': n},
        grid=(1,),
        buffers={'Vals': vals, 'Idx': indices, 'Z': z},
    )
    assert_close(results['Z'][:n], z_ref, atol=0, name="indirect_load")


def test_load_store_same_ptr():
    """In-place update: load from buffer, modify, store back."""
    n = 1024
    x = np.random.randn(n).astype(np.float32)
    z_ref = x * 2.0

    @triton.jit
    def kernel(X, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        val = tl.load(X + offs, mask=mask)
        tl.store(X + offs, val * 2.0, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x.copy()},
        scalars={'n_elements': n},
    )
    assert_close(results['X'][:n], z_ref, rtol=1e-5, name="load_store_same_ptr")


# ============================================================================
# Control Flow  (cf. test_if, test_if_else, test_for_iv, test_while)
# ============================================================================

def test_if_else():
    """If-else control flow."""
    @triton.jit
    def kernel(Cond, TrueVal, FalseVal, Out, BLOCK: tl.constexpr):
        cond = tl.load(Cond)
        if cond:
            val = tl.load(TrueVal)
        else:
            val = tl.load(FalseVal)
        tl.store(Out, val)

    cond_true = np.array([1], dtype=np.int32)
    cond_false = np.array([0], dtype=np.int32)
    true_val = np.array([42], dtype=np.int32)
    false_val = np.array([99], dtype=np.int32)
    out = np.zeros(1, dtype=np.int32)

    # Test TRUE branch
    results = compile_and_run(
        fn=kernel,
        signature={'Cond': '*i32', 'TrueVal': '*i32', 'FalseVal': '*i32',
                   'Out': '*i32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 1},
        grid=(1,),
        buffers={'Cond': cond_true, 'TrueVal': true_val,
                 'FalseVal': false_val, 'Out': out.copy()},
    )
    assert results['Out'][0] == 42, f"Expected 42, got {results['Out'][0]}"

    # Test FALSE branch
    results = compile_and_run(
        fn=kernel,
        signature={'Cond': '*i32', 'TrueVal': '*i32', 'FalseVal': '*i32',
                   'Out': '*i32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 1},
        grid=(1,),
        buffers={'Cond': cond_false, 'TrueVal': true_val,
                 'FalseVal': false_val, 'Out': out.copy()},
    )
    assert results['Out'][0] == 99, f"Expected 99, got {results['Out'][0]}"


def test_for_loop():
    """For loop: accumulate sum of range (static_range unrolls at compile time)."""
    @triton.jit
    def kernel(Out, N: tl.constexpr):
        acc = 0
        for i in tl.static_range(0, N):
            acc += 1
        tl.store(Out, acc)

    out = np.zeros(1, dtype=np.int32)
    N = 10
    results = compile_and_run(
        fn=kernel,
        signature={'Out': '*i32', 'N': 'constexpr'},
        constexprs={'N': N},
        grid=(1,),
        buffers={'Out': out},
    )
    assert results['Out'][0] == N, f"Expected {N}, got {results['Out'][0]}"


def test_for_loop_accumulate():
    """For loop with runtime bounds: sum elements of an array."""
    n = 256
    x = np.random.randn(n).astype(np.float32)
    z = np.zeros(1, dtype=np.float32)
    z_ref = np.sum(x)

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        acc = 0.0
        for i in tl.static_range(0, N):
            acc += tl.load(X + i)
        tl.store(Z, acc)

    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'N': 'constexpr'},
        constexprs={'N': n},
        grid=(1,),
        buffers={'X': x, 'Z': z},
    )
    assert_close(results['Z'][:1],
                 np.array([z_ref], dtype=np.float32),
                 rtol=0.02, name="for_accumulate")


def test_nested_for():
    """Nested for loops (static_range unrolls at compile time)."""
    @triton.jit
    def kernel(Out, M: tl.constexpr, N: tl.constexpr):
        acc = 0
        for i in tl.static_range(0, M):
            for j in tl.static_range(0, N):
                acc += 1
        tl.store(Out, acc)

    out = np.zeros(1, dtype=np.int32)
    M, N = 5, 7
    results = compile_and_run(
        fn=kernel,
        signature={'Out': '*i32', 'M': 'constexpr', 'N': 'constexpr'},
        constexprs={'M': M, 'N': N},
        grid=(1,),
        buffers={'Out': out},
    )
    assert results['Out'][0] == M * N, f"Expected {M * N}, got {results['Out'][0]}"


def test_while_loop():
    """While loop simulation via for loop with conditional break."""
    @triton.jit
    def kernel(Out, LIMIT: tl.constexpr):
        # Count up until reaching LIMIT (using static_range as an upper bound)
        acc = 0
        for _ in tl.static_range(0, 100):
            if acc < LIMIT:
                acc += 1
        tl.store(Out, acc)

    out = np.zeros(1, dtype=np.int32)
    LIMIT = 37
    results = compile_and_run(
        fn=kernel,
        signature={'Out': '*i32', 'LIMIT': 'constexpr'},
        constexprs={'LIMIT': LIMIT},
        grid=(1,),
        buffers={'Out': out},
    )
    assert results['Out'][0] == LIMIT, f"Expected {LIMIT}, got {results['Out'][0]}"


# ============================================================================
# Multiple Outputs  (cf. multi-output patterns in test_core.py)
# ============================================================================

def test_two_outputs():
    """Kernel writing to two separate output buffers."""
    n = 1024
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    out_sum = np.zeros(n, dtype=np.float32)
    out_prod = np.zeros(n, dtype=np.float32)

    @triton.jit
    def kernel(X, Y, OutSum, OutProd, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        y = tl.load(Y + offs, mask=mask)
        tl.store(OutSum + offs, x + y, mask=mask)
        tl.store(OutProd + offs, x * y, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Y': '*fp32', 'OutSum': '*fp32',
                   'OutProd': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Y': y, 'OutSum': out_sum, 'OutProd': out_prod},
        scalars={'n_elements': n},
    )
    assert_close(results['OutSum'][:n], x + y, name="two_out_sum")
    assert_close(results['OutProd'][:n], x * y, name="two_out_prod")


def test_statistics():
    """Compute mean and variance in a single kernel."""
    n = 128
    x = np.random.randn(n).astype(np.float32)
    out_mean = np.zeros(1, dtype=np.float32)
    out_var = np.zeros(1, dtype=np.float32)

    @triton.jit
    def kernel(X, OutMean, OutVar, N: tl.constexpr):
        vals = tl.load(X + tl.arange(0, N))
        mean = tl.sum(vals, axis=0) / N
        diff = vals - mean
        var = tl.sum(diff * diff, axis=0) / N
        tl.store(OutMean, mean)
        tl.store(OutVar, var)

    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'OutMean': '*fp32', 'OutVar': '*fp32',
                   'N': 'constexpr'},
        constexprs={'N': n},
        grid=(1,),
        buffers={'X': x, 'OutMean': out_mean, 'OutVar': out_var},
    )
    assert_close(results['OutMean'][:1], np.array([np.mean(x)], dtype=np.float32),
                 rtol=1e-3, name="stat_mean")
    assert_close(results['OutVar'][:1], np.array([np.var(x)], dtype=np.float32),
                 rtol=1e-2, name="stat_var")


# ============================================================================
# Vector Add  (cf. Tutorial 01 — basic end-to-end validation)
# ============================================================================

@pytest.mark.parametrize("n", [1, 100, 256, 1000, 1024, 8192])
def test_vector_add(n):
    """Vector addition: z = x + y for various sizes."""
    BLOCK_SIZE = 256
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    z_ref = x + y

    @triton.jit
    def kernel(X, Y, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        y = tl.load(Y + offs, mask=mask)
        tl.store(Z + offs, x + y, mask=mask)

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Y': '*fp32', 'Z': '*fp32',
                   'n_elements': 'i32', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK_SIZE},
        grid=grid,
        buffers={'X': x, 'Y': y, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, name=f"vadd_{n}")


# ============================================================================
# Elementwise Ops  (fused patterns)
# ============================================================================

def test_fused_multiply_add():
    """Fused multiply-add: z = a * b + c."""
    n = 1024
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    z_ref = a * b + c

    @triton.jit
    def kernel(A, B, C, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(A + offs, mask=mask)
        b = tl.load(B + offs, mask=mask)
        c = tl.load(C + offs, mask=mask)
        tl.store(Z + offs, a * b + c, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'A': '*fp32', 'B': '*fp32', 'C': '*fp32', 'Z': '*fp32',
                   'n_elements': 'i32', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'A': a, 'B': b, 'C': c, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, rtol=1e-5, name="fma")


def test_relu():
    """ReLU activation: z = max(0, x)."""
    n = 1024
    x = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    z_ref = np.maximum(0, x)

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        z = tl.where(x > 0, x, 0.0)
        tl.store(Z + offs, z, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, atol=0, name="relu")


def test_gelu():
    """Approximate GeLU activation."""
    n = 1024
    x = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    # Approximate GeLU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    z_ref = 0.5 * x * (1 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        # Approximate GeLU using exp-based tanh:
        # tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
        c = 0.7978845608028654  # sqrt(2/pi)
        a = c * (x + 0.044715 * x * x * x)
        e2a = tl.math.exp(2.0 * a)
        tanh_a = (e2a - 1.0) / (e2a + 1.0)
        z = 0.5 * x * (1.0 + tanh_a)
        tl.store(Z + offs, z, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, rtol=1e-4, atol=1e-4, name="gelu")


def test_softmax_row():
    """Row-wise softmax: numerically stable softmax across a row."""
    N = 128
    x = np.random.randn(N).astype(np.float32)
    z = np.zeros(N, dtype=np.float32)

    # Reference
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    z_ref = exp_x / np.sum(exp_x)

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        x_max = tl.max(x, axis=0)
        exp_x = tl.math.exp(x - x_max)
        sum_exp = tl.sum(exp_x, axis=0)
        z = exp_x / sum_exp
        tl.store(Z + offs, z)

    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'N': 'constexpr'},
        constexprs={'N': N},
        grid=(1,),
        buffers={'X': x, 'Z': z},
    )
    assert_close(results['Z'][:N], z_ref, rtol=1e-4, name="softmax")


# ============================================================================
# Program ID / num_programs  (cf. test_num_programs in test_core.py)
# ============================================================================

def test_program_id():
    """tl.program_id(axis=0) returns workgroup index."""
    n_groups = 16
    z = np.zeros(n_groups, dtype=np.int32)

    @triton.jit
    def kernel(Z, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        tl.store(Z + pid, pid)

    results = compile_and_run(
        fn=kernel,
        signature={'Z': '*i32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 1},
        grid=(n_groups,),
        buffers={'Z': z},
    )
    np.testing.assert_array_equal(
        np.sort(results['Z'][:n_groups]),
        np.arange(n_groups, dtype=np.int32),
        err_msg="program_id",
    )


def test_num_programs():
    """tl.num_programs(axis=0) returns grid size."""
    n_groups = 8
    z = np.zeros(n_groups, dtype=np.int32)

    @triton.jit
    def kernel(Z, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        num = tl.num_programs(0)
        tl.store(Z + pid, num)

    results = compile_and_run(
        fn=kernel,
        signature={'Z': '*i32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 1},
        grid=(n_groups,),
        buffers={'Z': z},
    )
    expected = np.full(n_groups, n_groups, dtype=np.int32)
    np.testing.assert_array_equal(results['Z'][:n_groups], expected,
                                  err_msg="num_programs")


# ============================================================================
# Constexpr  (cf. test_constexpr in test_core.py)
# ============================================================================

def test_constexpr_block_size():
    """Constexpr parameter controlling block size."""
    for block_size in [64, 128, 256]:
        z = np.zeros(block_size, dtype=np.float32)

        @triton.jit
        def kernel(Z, BLOCK_SIZE: tl.constexpr):
            offs = tl.arange(0, BLOCK_SIZE)
            tl.store(Z + offs, offs.to(tl.float32))

        results = compile_and_run(
            fn=kernel,
            signature={'Z': '*fp32', 'BLOCK_SIZE': 'constexpr'},
            constexprs={'BLOCK_SIZE': block_size},
            grid=(1,),
            buffers={'Z': z},
        )
        z_ref = np.arange(block_size, dtype=np.float32)
        assert_close(results['Z'][:block_size], z_ref, atol=0,
                     name=f"constexpr_{block_size}")


def test_multiple_constexprs():
    """Multiple constexpr parameters."""
    M, N = 4, 128  # N must match default workgroup size
    z = np.zeros(M * N, dtype=np.float32)

    @triton.jit
    def kernel(Z, M: tl.constexpr, N: tl.constexpr):
        for i in tl.static_range(0, M):
            offs = tl.arange(0, N)
            tl.store(Z + i * N + offs, (i * N + offs).to(tl.float32))

    results = compile_and_run(
        fn=kernel,
        signature={'Z': '*fp32', 'M': 'constexpr', 'N': 'constexpr'},
        constexprs={'M': M, 'N': N},
        grid=(1,),
        buffers={'Z': z},
    )
    z_ref = np.arange(M * N, dtype=np.float32)
    assert_close(results['Z'][:M * N], z_ref, atol=0, name="multi_constexpr")


# ============================================================================
# Integer-specific operations
# ============================================================================

def test_int_add():
    """Integer addition correctness."""
    n = 1024
    a = np.random.randint(-1000, 1000, n).astype(np.int32)
    b = np.random.randint(-1000, 1000, n).astype(np.int32)
    c = np.zeros(n, dtype=np.int32)
    c_ref = a + b

    @triton.jit
    def kernel(A, B, C, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(A + offs, mask=mask)
        b = tl.load(B + offs, mask=mask)
        tl.store(C + offs, a + b, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'A': '*i32', 'B': '*i32', 'C': '*i32',
                   'n_elements': 'i32', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'A': a, 'B': b, 'C': c},
        scalars={'n_elements': n},
    )
    np.testing.assert_array_equal(results['C'][:n], c_ref, err_msg="int_add")


def test_int_mul():
    """Integer multiplication."""
    n = 1024
    a = np.random.randint(-100, 100, n).astype(np.int32)
    b = np.random.randint(-100, 100, n).astype(np.int32)
    c = np.zeros(n, dtype=np.int32)
    c_ref = a * b

    @triton.jit
    def kernel(A, B, C, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(A + offs, mask=mask)
        b = tl.load(B + offs, mask=mask)
        tl.store(C + offs, a * b, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'A': '*i32', 'B': '*i32', 'C': '*i32',
                   'n_elements': 'i32', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'A': a, 'B': b, 'C': c},
        scalars={'n_elements': n},
    )
    np.testing.assert_array_equal(results['C'][:n], c_ref, err_msg="int_mul")


# ============================================================================
# Real-world Patterns (matching CUDA test_core.py functional patterns)
# ============================================================================

def test_vector_norm():
    """L2 norm of a vector: sqrt(sum(x*x))."""
    n = 128
    x = np.random.randn(n).astype(np.float32)
    z = np.zeros(1, dtype=np.float32)
    z_ref = np.sqrt(np.sum(x * x))

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        vals = tl.load(X + tl.arange(0, N))
        sq = vals * vals
        norm = tl.math.sqrt(tl.sum(sq, axis=0))
        tl.store(Z, norm)

    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'N': 'constexpr'},
        constexprs={'N': n},
        grid=(1,),
        buffers={'X': x, 'Z': z},
    )
    assert_close(results['Z'][:1], np.array([z_ref], dtype=np.float32),
                 rtol=1e-3, name="vector_norm")


def test_rmsnorm():
    """RMS normalization."""
    n = 128
    x = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x) + 1e-6)
    z_ref = x / rms

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        ms = tl.sum(x * x, axis=0) / N
        rms = tl.math.sqrt(ms + 1e-6)
        z = x / rms
        tl.store(Z + offs, z)

    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'N': 'constexpr'},
        constexprs={'N': n},
        grid=(1,),
        buffers={'X': x, 'Z': z},
    )
    assert_close(results['Z'][:n], z_ref, rtol=1e-3, name="rmsnorm")


def test_cross_entropy_loss():
    """Cross-entropy loss for a single sample."""
    n_classes = 128
    logits = np.random.randn(n_classes).astype(np.float32)
    labels = np.zeros(n_classes, dtype=np.float32)
    target = np.random.randint(0, n_classes)
    labels[target] = 1.0
    out = np.zeros(1, dtype=np.float32)

    # Reference: -sum(labels * log_softmax(logits))
    max_logit = np.max(logits)
    log_sum_exp = np.log(np.sum(np.exp(logits - max_logit))) + max_logit
    log_softmax = logits - log_sum_exp
    z_ref = -np.sum(labels * log_softmax)

    @triton.jit
    def kernel(Logits, Labels, Out, N: tl.constexpr):
        offs = tl.arange(0, N)
        logits = tl.load(Logits + offs)
        labels = tl.load(Labels + offs)
        max_l = tl.max(logits, axis=0)
        log_sum_exp = tl.math.log(tl.sum(tl.math.exp(logits - max_l), axis=0)) + max_l
        log_softmax = logits - log_sum_exp
        loss = -tl.sum(labels * log_softmax, axis=0)
        tl.store(Out, loss)

    results = compile_and_run(
        fn=kernel,
        signature={'Logits': '*fp32', 'Labels': '*fp32', 'Out': '*fp32',
                   'N': 'constexpr'},
        constexprs={'N': n_classes},
        grid=(1,),
        buffers={'Logits': logits, 'Labels': labels, 'Out': out},
    )
    assert_close(results['Out'][:1], np.array([z_ref], dtype=np.float32),
                 rtol=1e-3, name="cross_entropy")


# ============================================================================
# Scalar Operations
# ============================================================================

def test_scalar_multiply():
    """Multiply array by a scalar constant."""
    n = 1024
    x = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    z_ref = x * 2.5

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        tl.store(Z + offs, x * 2.5, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, name="scalar_mul")


def test_scalar_param():
    """Kernel receives scalar parameter (non-pointer, non-constexpr)."""
    n = 1024
    x = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    scale = 7
    z_ref = x * scale

    @triton.jit
    def kernel(X, Z, scale, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        tl.store(Z + offs, x * scale, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'scale': 'i32',
                   'n_elements': 'i32', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'scale': scale, 'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, rtol=1e-5, name="scalar_param")


# ============================================================================
# Clamp / Min / Max  (cf. test_clamp, test_propagate_nan in test_core.py)
# ============================================================================

def test_clamp():
    """Clamp values to [lo, hi] range."""
    n = 1024
    x = (np.random.randn(n) * 10).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    lo, hi = -2.0, 3.0
    z_ref = np.clip(x, lo, hi)

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        z = tl.where(x < -2.0, -2.0, tl.where(x > 3.0, 3.0, x))
        tl.store(Z + offs, z, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, atol=0, name="clamp")


def test_maximum():
    """Element-wise maximum of two arrays."""
    n = 1024
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    z_ref = np.maximum(a, b)

    @triton.jit
    def kernel(A, B, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(A + offs, mask=mask)
        b = tl.load(B + offs, mask=mask)
        z = tl.where(a > b, a, b)
        tl.store(Z + offs, z, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'A': '*fp32', 'B': '*fp32', 'Z': '*fp32',
                   'n_elements': 'i32', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'A': a, 'B': b, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, atol=0, name="maximum")


def test_minimum():
    """Element-wise minimum of two arrays."""
    n = 1024
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    z_ref = np.minimum(a, b)

    @triton.jit
    def kernel(A, B, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(A + offs, mask=mask)
        b = tl.load(B + offs, mask=mask)
        z = tl.where(a < b, a, b)
        tl.store(Z + offs, z, mask=mask)

    grid = ((n + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'A': '*fp32', 'B': '*fp32', 'Z': '*fp32',
                   'n_elements': 'i32', 'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'A': a, 'B': b, 'Z': z},
        scalars={'n_elements': n},
    )
    assert_close(results['Z'][:n], z_ref, atol=0, name="minimum")


# ============================================================================
# WGSL Translation validation  (verify shader structure)
# ============================================================================

def test_wgsl_shader_structure():
    """Verify compiled WGSL has correct WebGPU shader structure."""
    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        tl.store(Z + offs, tl.load(X + offs, mask=mask), mask=mask)

    k = compile_webgpu(
        kernel,
        {'X': '*fp32', 'Z': '*fp32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
        {'BLOCK_SIZE': 256},
    )

    sig = {'X': '*fp32', 'Z': '*fp32', 'n': 'i32'}
    result = translate_llvm_to_wgsl(k.asm['llir'], sig)

    wgsl = result.wgsl
    assert '@compute' in wgsl, "Missing @compute attribute"
    assert '@workgroup_size' in wgsl, "Missing @workgroup_size"
    assert 'fn main(' in wgsl, "Missing main function"
    assert 'workgroup_id' in wgsl, "Missing workgroup_id builtin"
    assert 'local_invocation_id' in wgsl, "Missing local_invocation_id builtin"
    assert len(result.buffer_bindings) >= 2, "Expected at least 2 buffer bindings"
    assert len(result.param_fields) >= 1, "Expected scalar params"


def test_wgsl_in_compiled_asm():
    """Verify WGSL stage is included in compiled kernel's asm dict."""
    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        tl.store(Z + offs, tl.load(X + offs, mask=mask), mask=mask)

    k = compile_webgpu(
        kernel,
        {'X': '*fp32', 'Z': '*fp32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
        {'BLOCK_SIZE': 256},
    )

    assert 'wgsl' in k.asm, "WGSL stage should be in compiled asm dict"
    assert '@compute' in k.asm['wgsl'], "WGSL should contain @compute"


def test_all_compilation_stages():
    """Verify all compilation stages produce output."""
    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        tl.store(Z + offs, tl.load(X + offs, mask=mask), mask=mask)

    k = compile_webgpu(
        kernel,
        {'X': '*fp32', 'Z': '*fp32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
        {'BLOCK_SIZE': 256},
    )

    # All stages should be present
    for stage in ['ttir', 'ttgir', 'llir', 'spv', 'wgsl']:
        assert stage in k.asm, f"Missing compilation stage: {stage}"
        assert k.asm[stage], f"Empty output for stage: {stage}"


# ============================================================================
# Performance Benchmark
# ============================================================================

def test_vector_add_bandwidth():
    """Measure effective bandwidth for vector addition."""
    N = 1 << 20  # 1M elements
    BLOCK_SIZE = 256
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    expected = x + y

    @triton.jit
    def kernel(X, Y, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(X + offs, mask=mask)
        y = tl.load(Y + offs, mask=mask)
        tl.store(Z + offs, x + y, mask=mask)

    sig = {'X': '*fp32', 'Y': '*fp32', 'Z': '*fp32',
           'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    sig_no_ce = {k: v for k, v in sig.items() if v != 'constexpr'}
    k = compile_webgpu(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE})
    result = translate_llvm_to_wgsl(k.asm['llir'], sig_no_ce, 4, 32)
    runner = get_runner()
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Warmup
    runner.run_kernel(
        wgsl_code=result.wgsl, buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields, workgroup_size=result.workgroup_size,
        grid=grid, buffers={'X': x, 'Y': y, 'Z': out}, scalars={'n': N},
    )

    # Timed run
    num_iters = 5
    t0 = time.perf_counter()
    for _ in range(num_iters):
        outputs = runner.run_kernel(
            wgsl_code=result.wgsl, buffer_bindings=result.buffer_bindings,
            param_fields=result.param_fields, workgroup_size=result.workgroup_size,
            grid=grid, buffers={'X': x, 'Y': y, 'Z': out}, scalars={'n': N},
        )
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) / num_iters * 1000
    bytes_transferred = 3 * N * 4
    bandwidth_gb_s = bytes_transferred / (elapsed_ms / 1000) / 1e9
    print(f"\n  Vector Add (N={N:,}): {elapsed_ms:.2f} ms, "
          f"{bandwidth_gb_s:.1f} GB/s effective bandwidth")

    assert_close(outputs['Z'][:N], expected, name="perf_add")


# ============================================================================
# Atomic Operations
# ============================================================================

def test_atomic_add_int():
    """Test integer atomic add across threads."""
    N = 256
    BLOCK_SIZE = 256
    x = np.ones(N, dtype=np.int32)
    out = np.zeros(N, dtype=np.int32)

    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(X + offs, mask=mask)
        tl.atomic_add(Z + offs, x, mask=mask)

    sig = {'X': '*i32', 'Z': '*i32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    outputs = compile_and_run(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE},
                              grid=((N + BLOCK_SIZE - 1) // BLOCK_SIZE,),
                              buffers={'X': x, 'Z': out},
                              scalars={'n': N})
    expected = x  # 0 + 1 = 1 for each element
    np.testing.assert_array_equal(outputs['Z'][:N], expected)


def test_atomic_add_int_accumulate():
    """Test integer atomic add with multiple workgroups accumulating."""
    N = 64
    BLOCK_SIZE = 64
    NUM_WG = 4
    NUM_WARPS = 2  # workgroup_size = 64 = BLOCK_SIZE, so 1 thread per element
    # Each workgroup adds 1 to every element, so result should be NUM_WG
    x = np.ones(N, dtype=np.int32)
    out = np.zeros(N, dtype=np.int32)

    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(X + offs, mask=mask)
        tl.atomic_add(Z + offs, x, mask=mask)

    sig = {'X': '*i32', 'Z': '*i32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    outputs = compile_and_run(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE},
                              grid=(NUM_WG,),
                              buffers={'X': x, 'Z': out},
                              scalars={'n': N},
                              num_warps=NUM_WARPS)
    expected = np.full(N, NUM_WG, dtype=np.int32)
    np.testing.assert_array_equal(outputs['Z'][:N], expected)


def test_atomic_add_float():
    """Test float atomic add across threads."""
    N = 256
    BLOCK_SIZE = 256
    x = np.ones(N, dtype=np.float32) * 1.5
    out = np.zeros(N, dtype=np.float32)

    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(X + offs, mask=mask)
        tl.atomic_add(Z + offs, x, mask=mask)

    sig = {'X': '*fp32', 'Z': '*fp32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    outputs = compile_and_run(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE},
                              grid=((N + BLOCK_SIZE - 1) // BLOCK_SIZE,),
                              buffers={'X': x, 'Z': out},
                              scalars={'n': N})
    expected = x  # 0.0 + 1.5 = 1.5
    assert_close(outputs['Z'][:N], expected, name="atomic_fadd")


def test_atomic_add_float_accumulate():
    """Test float atomic add with multiple workgroups accumulating."""
    N = 64
    BLOCK_SIZE = 64
    NUM_WG = 4
    NUM_WARPS = 2  # workgroup_size = 64 = BLOCK_SIZE, so 1 thread per element
    x = np.ones(N, dtype=np.float32)
    out = np.zeros(N, dtype=np.float32)

    @triton.jit
    def kernel(X, Z, n, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(X + offs, mask=mask)
        tl.atomic_add(Z + offs, x, mask=mask)

    sig = {'X': '*fp32', 'Z': '*fp32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
    outputs = compile_and_run(kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE},
                              grid=(NUM_WG,),
                              buffers={'X': x, 'Z': out},
                              scalars={'n': N},
                              num_warps=NUM_WARPS)
    expected = np.full(N, float(NUM_WG), dtype=np.float32)
    assert_close(outputs['Z'][:N], expected, name="atomic_fadd_accum")


# ============================================================================
# GEMM / tl.dot tests
# ============================================================================

@triton.jit
def _gemm_kernel(A, B, C, M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs + k)
        b = tl.load(b_ptrs + k * N)
        acc += tl.dot(a, b)
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc)


_GEMM_SIG = {
    'A': '*fp32', 'B': '*fp32', 'C': '*fp32',
    'M': 'i32', 'N': 'i32', 'K': 'i32',
    'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr',
}


def _run_gemm(M, N, K, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16, seed=42):
    """Helper: compile & run GEMM, return (gpu_result, expected)."""
    np.random.seed(seed)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    expected = A @ B
    constexprs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K}
    outputs = compile_and_run(
        _gemm_kernel, _GEMM_SIG, constexprs,
        grid=(M // BLOCK_M, N // BLOCK_N, 1),
        buffers={'A': A.ravel(), 'B': B.ravel(), 'C': C.ravel()},
        scalars={'M': M, 'N': N, 'K': K},
    )
    return outputs['C'].reshape(M, N), expected


def test_gemm_single_tile():
    """GEMM with one 16x16 tile — single workgroup, single k-iteration."""
    gpu, ref = _run_gemm(16, 16, 16)
    assert_close(gpu, ref, atol=1e-4, name="gemm_single_tile")


def test_gemm_k_loop():
    """GEMM with K>BLOCK_K — exercises the k-loop."""
    gpu, ref = _run_gemm(16, 16, 64)
    assert_close(gpu, ref, atol=1e-3, name="gemm_k_loop")


def test_gemm_2d_grid():
    """GEMM with multi-tile 2D grid dispatch."""
    gpu, ref = _run_gemm(64, 64, 64)
    assert_close(gpu, ref, atol=1e-3, name="gemm_2d_grid")


@pytest.mark.parametrize("M,N,K", [
    (32, 32, 16),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
])
def test_gemm_sizes(M, N, K):
    """GEMM correctness across various matrix sizes."""
    gpu, ref = _run_gemm(M, N, K)
    assert_close(gpu, ref, atol=1e-2, name=f"gemm_{M}x{N}x{K}")


def test_gemm_non_square():
    """GEMM with non-square matrices."""
    gpu, ref = _run_gemm(32, 64, 48, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
    assert_close(gpu, ref, atol=1e-2, name="gemm_non_square")


# ============================================================================
# LayerNorm Tests
# ============================================================================

@triton.jit
def _layer_norm_kernel(X, Y, W, B, Mean, Rstd, stride, N, eps,
                       BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xmean = x - mean
    var = tl.sum(xmean * xmean, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = xmean * rstd
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xhat * w + b
    tl.store(Y + row * stride + cols, y, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


def _run_layer_norm(M, N, BLOCK_SIZE, num_warps):
    """Run LayerNorm kernel and return (gpu_outputs, ref_outputs)."""
    np.random.seed(42)
    X = np.random.randn(M, N).astype(np.float32)
    W = np.random.randn(N).astype(np.float32)
    B = np.random.randn(N).astype(np.float32)
    eps = 1e-5

    sig = {'X': '*fp32', 'Y': '*fp32', 'W': '*fp32', 'B': '*fp32',
           'Mean': '*fp32', 'Rstd': '*fp32', 'stride': 'i32', 'N': 'i32',
           'eps': 'fp32', 'BLOCK_SIZE': 'constexpr'}
    constexprs = {'BLOCK_SIZE': BLOCK_SIZE}
    sig_no_ce = {k: v for k, v in sig.items() if v != 'constexpr'}

    src = ASTSource(fn=_layer_norm_kernel, signature=sig, constexprs=constexprs)
    compiled = triton.compile(src, target=WEBGPU_TARGET)
    result = translate_llvm_to_wgsl(
        compiled.asm['llir'], sig_no_ce,
        num_warps=num_warps, warp_size=32)

    runner = DawnRunner()
    out = runner.run_kernel(
        wgsl_code=result.wgsl,
        buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields,
        workgroup_size=result.workgroup_size,
        grid=(M,),
        buffers={
            'X': X.ravel(), 'Y': np.zeros(M * N, dtype=np.float32),
            'W': W, 'B': B,
            'Mean': np.zeros(M, dtype=np.float32),
            'Rstd': np.zeros(M, dtype=np.float32),
        },
        scalars={'stride': N, 'N': N, 'eps': eps})

    # Reference
    x_mean = X.mean(axis=1)
    x_var = X.var(axis=1)
    x_rstd = 1.0 / np.sqrt(x_var + eps)
    x_hat = (X - x_mean[:, None]) * x_rstd[:, None]
    y_ref = x_hat * W[None, :] + B[None, :]

    return {
        'Mean': out['Mean'][:M],
        'Rstd': out['Rstd'][:M],
        'Y': out['Y'].reshape(M, N),
    }, {
        'Mean': x_mean,
        'Rstd': x_rstd,
        'Y': y_ref,
    }


def test_layer_norm_basic():
    """LayerNorm: basic M=4, N=32, single warp."""
    gpu, ref = _run_layer_norm(M=4, N=32, BLOCK_SIZE=32, num_warps=1)
    assert_close(gpu['Mean'], ref['Mean'], atol=1e-3, name="ln_mean")
    assert_close(gpu['Rstd'], ref['Rstd'], atol=1e-3, name="ln_rstd")
    assert_close(gpu['Y'], ref['Y'], atol=1e-2, name="ln_Y")


def test_layer_norm_multi_warp():
    """LayerNorm: M=4, N=64, 2 warps."""
    gpu, ref = _run_layer_norm(M=4, N=64, BLOCK_SIZE=64, num_warps=2)
    assert_close(gpu['Mean'], ref['Mean'], atol=1e-3, name="ln_mean_2w")
    assert_close(gpu['Rstd'], ref['Rstd'], atol=1e-3, name="ln_rstd_2w")
    assert_close(gpu['Y'], ref['Y'], atol=1e-2, name="ln_Y_2w")


def test_layer_norm_4_warps():
    """LayerNorm: M=4, N=64, 4 warps."""
    gpu, ref = _run_layer_norm(M=4, N=64, BLOCK_SIZE=64, num_warps=4)
    assert_close(gpu['Mean'], ref['Mean'], atol=1e-3, name="ln_mean_4w")
    assert_close(gpu['Rstd'], ref['Rstd'], atol=1e-3, name="ln_rstd_4w")
    assert_close(gpu['Y'], ref['Y'], atol=1e-2, name="ln_Y_4w")


@pytest.mark.parametrize("M,N,BS,nw", [
    (8, 128, 128, 4),
    (4, 256, 256, 4),
    (16, 64, 64, 2),
])
def test_layer_norm_sizes(M, N, BS, nw):
    """LayerNorm: parametrized sizes."""
    gpu, ref = _run_layer_norm(M=M, N=N, BLOCK_SIZE=BS, num_warps=nw)
    assert_close(gpu['Mean'], ref['Mean'], atol=1e-3,
                 name=f"ln_mean_{M}x{N}")
    assert_close(gpu['Rstd'], ref['Rstd'], atol=1e-3,
                 name=f"ln_rstd_{M}x{N}")
    assert_close(gpu['Y'], ref['Y'], atol=1e-2,
                 name=f"ln_Y_{M}x{N}")


# ============================================================================
# Softmax Tests
# ============================================================================

@triton.jit
def _softmax_kernel(X, Y, stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask,
                other=float('-inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    sum_exp = tl.sum(exp_x, axis=0)
    y = exp_x / sum_exp
    tl.store(Y + row * stride + cols, y, mask=mask)


def _run_softmax(M, N, BLOCK_SIZE, num_warps):
    """Run softmax kernel and return (gpu_Y, ref_Y)."""
    np.random.seed(42)
    X = np.random.randn(M, N).astype(np.float32)

    sig = {'X': '*fp32', 'Y': '*fp32', 'stride': 'i32', 'N': 'i32',
           'BLOCK_SIZE': 'constexpr'}
    constexprs = {'BLOCK_SIZE': BLOCK_SIZE}
    sig_no_ce = {k: v for k, v in sig.items() if v != 'constexpr'}

    src = ASTSource(fn=_softmax_kernel, signature=sig, constexprs=constexprs)
    compiled = triton.compile(src, target=WEBGPU_TARGET)
    result = translate_llvm_to_wgsl(
        compiled.asm['llir'], sig_no_ce,
        num_warps=num_warps, warp_size=32)

    runner = DawnRunner()
    out = runner.run_kernel(
        wgsl_code=result.wgsl,
        buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields,
        workgroup_size=result.workgroup_size,
        grid=(M,),
        buffers={'X': X.ravel(), 'Y': np.zeros(M * N, dtype=np.float32)},
        scalars={'stride': N, 'N': N})

    Y_gpu = out['Y'].reshape(M, N)
    exp_x = np.exp(X - X.max(axis=1, keepdims=True))
    y_ref = exp_x / exp_x.sum(axis=1, keepdims=True)
    return Y_gpu, y_ref


def test_softmax_basic():
    """Softmax: single warp, M=4, N=32."""
    gpu, ref = _run_softmax(M=4, N=32, BLOCK_SIZE=32, num_warps=1)
    assert_close(gpu, ref, atol=1e-5, name="softmax_basic")


def test_softmax_multi_warp():
    """Softmax: 2 warps, M=4, N=64."""
    gpu, ref = _run_softmax(M=4, N=64, BLOCK_SIZE=64, num_warps=2)
    assert_close(gpu, ref, atol=1e-5, name="softmax_2w")


@pytest.mark.parametrize("M,N,BS,nw", [
    (4, 64, 64, 4),
    (8, 128, 128, 4),
    (4, 256, 256, 4),
])
def test_softmax_sizes(M, N, BS, nw):
    """Softmax: parametrized sizes."""
    gpu, ref = _run_softmax(M=M, N=N, BLOCK_SIZE=BS, num_warps=nw)
    assert_close(gpu, ref, atol=1e-5, name=f"softmax_{M}x{N}")


# ============================================================================
# RoPE (Rotary Positional Embedding) Tests
# ============================================================================

@triton.jit
def _rope_kernel(X, COS, SIN, Y, stride_seq, stride_head,
                 HALF_DIM: tl.constexpr):
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)
    cols = tl.arange(0, HALF_DIM)
    base = pid_seq * stride_seq + pid_head * stride_head
    x0 = tl.load(X + base + cols).to(tl.float32)
    x1 = tl.load(X + base + HALF_DIM + cols).to(tl.float32)
    cos_val = tl.load(COS + pid_seq * HALF_DIM + cols).to(tl.float32)
    sin_val = tl.load(SIN + pid_seq * HALF_DIM + cols).to(tl.float32)
    y0 = x0 * cos_val - x1 * sin_val
    y1 = x0 * sin_val + x1 * cos_val
    tl.store(Y + base + cols, y0)
    tl.store(Y + base + HALF_DIM + cols, y1)


def _run_rope(seq_len, n_heads, head_dim):
    """Run RoPE kernel and return (gpu_Y, ref_Y)."""
    half_dim = head_dim // 2
    np.random.seed(42)
    X = np.random.randn(seq_len, n_heads, head_dim).astype(np.float32)
    pos = np.arange(seq_len).reshape(-1, 1)
    dim = np.arange(half_dim).reshape(1, -1)
    theta = pos / (10000.0 ** (2 * dim / head_dim))
    COS = np.cos(theta).astype(np.float32)
    SIN = np.sin(theta).astype(np.float32)

    sig = {'X': '*fp32', 'COS': '*fp32', 'SIN': '*fp32', 'Y': '*fp32',
           'stride_seq': 'i32', 'stride_head': 'i32', 'HALF_DIM': 'constexpr'}
    constexprs = {'HALF_DIM': half_dim}
    sig_no_ce = {k: v for k, v in sig.items() if v != 'constexpr'}
    src = ASTSource(fn=_rope_kernel, signature=sig, constexprs=constexprs)
    compiled = triton.compile(src, target=WEBGPU_TARGET)
    result = translate_llvm_to_wgsl(
        compiled.asm['llir'], sig_no_ce, num_warps=1, warp_size=32)

    runner = DawnRunner()
    stride_seq = n_heads * head_dim
    stride_head = head_dim
    out = runner.run_kernel(
        wgsl_code=result.wgsl, buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields, workgroup_size=result.workgroup_size,
        grid=(seq_len, n_heads),
        buffers={'X': X.ravel(), 'COS': COS.ravel(), 'SIN': SIN.ravel(),
                 'Y': np.zeros_like(X).ravel()},
        scalars={'stride_seq': stride_seq, 'stride_head': stride_head})

    Y_gpu = out['Y'].reshape(seq_len, n_heads, head_dim)
    x0 = X[..., :half_dim]
    x1 = X[..., half_dim:]
    cos_r = COS[:, np.newaxis, :]
    sin_r = SIN[:, np.newaxis, :]
    Y_ref = np.concatenate(
        [x0 * cos_r - x1 * sin_r, x0 * sin_r + x1 * cos_r], axis=-1)
    return Y_gpu, Y_ref


def test_rope_basic():
    """RoPE: seq=8, heads=4, head_dim=32."""
    gpu, ref = _run_rope(seq_len=8, n_heads=4, head_dim=32)
    assert_close(gpu, ref, atol=1e-5, name="rope_basic")


def test_rope_large():
    """RoPE: seq=16, heads=8, head_dim=64."""
    gpu, ref = _run_rope(seq_len=16, n_heads=8, head_dim=64)
    assert_close(gpu, ref, atol=1e-5, name="rope_large")


# ============================================================================
# Cross-Entropy Loss Tests
# ============================================================================

@triton.jit
def _cross_entropy_kernel(Logits, Targets, Losses, n_classes, stride,
                          BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_classes
    logits = tl.load(Logits + row * stride + cols, mask=mask,
                     other=float('-inf')).to(tl.float32)
    max_logit = tl.max(logits, axis=0)
    shifted = logits - max_logit
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_logit
    target_idx = tl.load(Targets + row)
    target_logit = tl.load(Logits + row * stride + target_idx).to(tl.float32)
    loss = log_sum_exp - target_logit
    tl.store(Losses + row, loss)


def _run_cross_entropy(batch, n_classes, block_size, num_warps=1):
    """Run cross-entropy kernel and return (gpu_losses, ref_losses)."""
    np.random.seed(42)
    logits = np.random.randn(batch, n_classes).astype(np.float32)
    targets = np.random.randint(0, n_classes, size=batch).astype(np.int32)

    sig = {'Logits': '*fp32', 'Targets': '*i32', 'Losses': '*fp32',
           'n_classes': 'i32', 'stride': 'i32', 'BLOCK_SIZE': 'constexpr'}
    constexprs = {'BLOCK_SIZE': block_size}
    sig_no_ce = {k: v for k, v in sig.items() if v != 'constexpr'}
    src = ASTSource(fn=_cross_entropy_kernel, signature=sig,
                    constexprs=constexprs)
    compiled = triton.compile(src, target=WEBGPU_TARGET)
    result = translate_llvm_to_wgsl(
        compiled.asm['llir'], sig_no_ce,
        num_warps=num_warps, warp_size=32)

    runner = DawnRunner()
    out = runner.run_kernel(
        wgsl_code=result.wgsl, buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields, workgroup_size=result.workgroup_size,
        grid=(batch,),
        buffers={'Logits': logits.ravel(), 'Targets': targets,
                 'Losses': np.zeros(batch, dtype=np.float32)},
        scalars={'n_classes': n_classes, 'stride': n_classes})

    losses_gpu = out['Losses'][:batch]
    # Reference: -log(softmax(logits)[target])
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1))
    losses_ref = (log_sum_exp - shifted[np.arange(batch), targets]).astype(
        np.float32)
    return losses_gpu, losses_ref


def test_cross_entropy_basic():
    """Cross-entropy: batch=8, 32 classes."""
    gpu, ref = _run_cross_entropy(batch=8, n_classes=32, block_size=32)
    assert_close(gpu, ref, atol=1e-4, name="ce_basic")


def test_cross_entropy_large():
    """Cross-entropy: batch=16, 64 classes."""
    gpu, ref = _run_cross_entropy(batch=16, n_classes=64, block_size=64,
                                  num_warps=2)
    assert_close(gpu, ref, atol=1e-4, name="ce_large")


# ============================================================================
# Causal Attention Tests
# ============================================================================

@triton.jit
def _causal_attn_kernel(Q, K, V, Out,
                        stride_q, stride_k, stride_v, stride_o,
                        seq_len, scale,
                        BLOCK_HD: tl.constexpr):
    q_pos = tl.program_id(0)
    hd = tl.arange(0, BLOCK_HD)
    q = tl.load(Q + q_pos * stride_q + hd).to(tl.float32)
    acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
    m_prev = -1e9 + tl.zeros([1], dtype=tl.float32)
    l_prev = tl.zeros([1], dtype=tl.float32)
    for k_pos in range(q_pos + 1):
        k = tl.load(K + k_pos * stride_k + hd).to(tl.float32)
        score = tl.sum(q * k, axis=0) * scale
        m_new = tl.maximum(m_prev, score)
        exp_prev = tl.exp(m_prev - m_new)
        exp_score = tl.exp(score - m_new)
        l_new = l_prev * exp_prev + exp_score
        v = tl.load(V + k_pos * stride_v + hd).to(tl.float32)
        acc = acc * (l_prev * exp_prev / l_new) + v * (exp_score / l_new)
        m_prev = m_new
        l_prev = l_new
    tl.store(Out + q_pos * stride_o + hd, acc)


def _run_causal_attn(seq_len, head_dim):
    """Run causal attention kernel and return (gpu_out, ref_out)."""
    np.random.seed(42)
    Q = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    K = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    V = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    scale = float(1.0 / np.sqrt(head_dim))

    sig = {'Q': '*fp32', 'K': '*fp32', 'V': '*fp32', 'Out': '*fp32',
           'stride_q': 'i32', 'stride_k': 'i32',
           'stride_v': 'i32', 'stride_o': 'i32',
           'seq_len': 'i32', 'scale': 'fp32', 'BLOCK_HD': 'constexpr'}
    constexprs = {'BLOCK_HD': head_dim}
    sig_no_ce = {k: v for k, v in sig.items() if v != 'constexpr'}
    src = ASTSource(fn=_causal_attn_kernel, signature=sig,
                    constexprs=constexprs)
    compiled = triton.compile(src, target=WEBGPU_TARGET)
    result = translate_llvm_to_wgsl(
        compiled.asm['llir'], sig_no_ce, num_warps=1, warp_size=32)

    runner = DawnRunner()
    out = runner.run_kernel(
        wgsl_code=result.wgsl, buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields, workgroup_size=result.workgroup_size,
        grid=(seq_len,),
        buffers={'Q': Q.ravel(), 'K': K.ravel(), 'V': V.ravel(),
                 'Out': np.zeros_like(Q).ravel()},
        scalars={'stride_q': head_dim, 'stride_k': head_dim,
                 'stride_v': head_dim, 'stride_o': head_dim,
                 'seq_len': seq_len, 'scale': scale})

    Out_gpu = out['Out'].reshape(seq_len, head_dim)
    # Reference: masked softmax attention
    scores = Q @ K.T * scale
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores[mask] = -1e9
    exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
    attn = exp_s / exp_s.sum(axis=1, keepdims=True)
    Out_ref = attn @ V
    return Out_gpu, Out_ref


def test_causal_attn_basic():
    """Causal attention: seq=8, head_dim=16."""
    gpu, ref = _run_causal_attn(seq_len=8, head_dim=16)
    assert_close(gpu, ref, atol=1e-4, name="causal_attn_basic")


def test_causal_attn_large():
    """Causal attention: seq=16, head_dim=32."""
    gpu, ref = _run_causal_attn(seq_len=16, head_dim=32)
    assert_close(gpu, ref, atol=1e-4, name="causal_attn_large")


# ============================================================================
# Linear projection tests
# ============================================================================

def _run_linear(M, N, K, num_warps=4):
    """Test Y = X @ W^T + bias via per-element dot."""
    np.random.seed(42)
    x = np.random.randn(M, K).astype(np.float32) * 0.1
    w = np.random.randn(N, K).astype(np.float32) * 0.1
    bias = np.random.randn(N).astype(np.float32) * 0.01

    BLOCK_K = _next_pow2(K)
    # Pad if K < BLOCK_K
    if K < BLOCK_K:
        x_buf = np.zeros((M, BLOCK_K), dtype=np.float32)
        x_buf[:, :K] = x
        w_buf = np.zeros((N, BLOCK_K), dtype=np.float32)
        w_buf[:, :K] = w
    else:
        x_buf = x
        w_buf = w

    @triton.jit
    def kernel(X, W, Bias, Y, stride_x, stride_w, N_out,
               BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        col = tl.program_id(1)
        ks = tl.arange(0, BLOCK_K)
        xv = tl.load(X + row * stride_x + ks).to(tl.float32)
        wv = tl.load(W + col * stride_w + ks).to(tl.float32)
        dot = tl.sum(xv * wv, axis=0)
        b = tl.load(Bias + col).to(tl.float32)
        tl.store(Y + row * N_out + col, dot + b)

    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'W': '*fp32', 'Bias': '*fp32', 'Y': '*fp32',
                   'stride_x': 'i32', 'stride_w': 'i32', 'N_out': 'i32',
                   'BLOCK_K': 'constexpr'},
        constexprs={'BLOCK_K': BLOCK_K},
        grid=(M, N),
        buffers={'X': x_buf.ravel(), 'W': w_buf.ravel(),
                 'Bias': bias, 'Y': np.zeros(M * N, dtype=np.float32)},
        scalars={'stride_x': BLOCK_K, 'stride_w': BLOCK_K, 'N_out': N},
        num_warps=max(1, BLOCK_K // 32),
    )
    gpu = results['Y'].reshape(M, N)
    ref = (x @ w.T + bias).astype(np.float32)
    return gpu, ref


def test_linear_basic():
    """Linear projection: 4×16 @ 8×16^T + bias → 4×8."""
    gpu, ref = _run_linear(M=4, N=8, K=16)
    assert_close(gpu, ref, atol=1e-4, name="linear_basic")


def test_linear_large():
    """Linear projection: 8×64 @ 32×64^T + bias → 8×32."""
    gpu, ref = _run_linear(M=8, N=32, K=64)
    assert_close(gpu, ref, atol=1e-4, name="linear_large")


# ============================================================================
# GELU activation tests
# ============================================================================

def _run_gelu(N, BLOCK=128, num_warps=4):
    """Test approximate GELU activation."""
    np.random.seed(42)
    x = np.random.randn(N).astype(np.float32) * 2.0

    @triton.jit
    def kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask)
        c = 0.7978845608028654
        a = c * (x + 0.044715 * x * x * x)
        e2a = tl.math.exp(2.0 * a)
        tanh_a = (e2a - 1.0) / (e2a + 1.0)
        z = 0.5 * x * (1.0 + tanh_a)
        tl.store(Z + offs, z, mask=mask)

    grid = ((N + BLOCK - 1) // BLOCK,)
    results = compile_and_run(
        fn=kernel,
        signature={'X': '*fp32', 'Z': '*fp32', 'n_elements': 'i32',
                   'BLOCK_SIZE': 'constexpr'},
        constexprs={'BLOCK_SIZE': BLOCK},
        grid=grid,
        buffers={'X': x, 'Z': np.zeros(N, dtype=np.float32)},
        scalars={'n_elements': N},
        num_warps=num_warps,
    )
    gpu = results['Z'][:N]
    ref = (0.5 * x * (1 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))).astype(np.float32)
    return gpu, ref


def test_gelu_basic():
    """GELU activation: 64 elements."""
    gpu, ref = _run_gelu(N=64)
    assert_close(gpu, ref, atol=1e-4, name="gelu_basic")


def test_gelu_large():
    """GELU activation: 256 elements, 2 blocks."""
    gpu, ref = _run_gelu(N=256, BLOCK=128)
    assert_close(gpu, ref, atol=1e-4, name="gelu_large")


# ============================================================================
# GPT-2 pipeline verification test
# ============================================================================

def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def test_gpt2_pipeline():
    """End-to-end GPT-2 forward pass with random weights (2 layers, 64 embd).

    Verifies that all transformer components work together correctly:
    LayerNorm → Attention (QKV linear + causal attn + output proj) →
    residual → LayerNorm → MLP (fc + GELU + proj) → residual.
    """
    n_layer, n_head, n_embd, n_vocab = 2, 2, 64, 256
    head_dim = n_embd // n_head
    np.random.seed(42)

    # Build random weight dict
    weights = {}
    weights["wte"] = np.random.randn(n_vocab, n_embd).astype(np.float32) * 0.02
    weights["wpe"] = np.random.randn(128, n_embd).astype(np.float32) * 0.02
    weights["ln_f_w"] = np.ones(n_embd, dtype=np.float32)
    weights["ln_f_b"] = np.zeros(n_embd, dtype=np.float32)

    for i in range(n_layer):
        pfx = f"l{i}_"
        weights[pfx + "ln1_w"] = np.ones(n_embd, dtype=np.float32)
        weights[pfx + "ln1_b"] = np.zeros(n_embd, dtype=np.float32)
        weights[pfx + "ln2_w"] = np.ones(n_embd, dtype=np.float32)
        weights[pfx + "ln2_b"] = np.zeros(n_embd, dtype=np.float32)
        weights[pfx + "qkv_w"] = np.random.randn(3 * n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "qkv_b"] = np.zeros(3 * n_embd, dtype=np.float32)
        weights[pfx + "proj_w"] = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "proj_b"] = np.zeros(n_embd, dtype=np.float32)
        weights[pfx + "fc_w"] = np.random.randn(4 * n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "fc_b"] = np.zeros(4 * n_embd, dtype=np.float32)
        weights[pfx + "fc2_w"] = np.random.randn(n_embd, 4 * n_embd).astype(np.float32) * 0.02
        weights[pfx + "fc2_b"] = np.zeros(n_embd, dtype=np.float32)

    # Compile kernels
    BS = _next_pow2(n_embd)   # = 64
    BS4 = _next_pow2(4 * n_embd)  # = 256
    nw = max(1, BS // 32)
    nw4 = max(1, BS4 // 32)

    @triton.jit
    def ln_k(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        xm = x - mean
        var = tl.sum(xm * xm, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        y = xm * rstd
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + row * stride + cols, y * w + b, mask=mask)
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)

    @triton.jit
    def lin_k(X, W, Bias, Y, stride_x, stride_w, N_out, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        col = tl.program_id(1)
        ks = tl.arange(0, BLOCK_K)
        xv = tl.load(X + row * stride_x + ks).to(tl.float32)
        wv = tl.load(W + col * stride_w + ks).to(tl.float32)
        dot = tl.sum(xv * wv, axis=0)
        b = tl.load(Bias + col).to(tl.float32)
        tl.store(Y + row * N_out + col, dot + b)

    @triton.jit
    def attn_k(Q, K, V, Out, stride_q, stride_k, stride_v, stride_o,
               seq_len, scale, BLOCK_HD: tl.constexpr):
        q_pos = tl.program_id(0)
        hd = tl.arange(0, BLOCK_HD)
        q = tl.load(Q + q_pos * stride_q + hd).to(tl.float32)
        acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
        m_prev = -1e9 + tl.zeros([1], dtype=tl.float32)
        l_prev = tl.zeros([1], dtype=tl.float32)
        for k_pos in range(q_pos + 1):
            k = tl.load(K + k_pos * stride_k + hd).to(tl.float32)
            score = tl.sum(q * k, axis=0) * scale
            m_new = tl.maximum(m_prev, score)
            exp_prev = tl.exp(m_prev - m_new)
            exp_score = tl.exp(score - m_new)
            l_new = l_prev * exp_prev + exp_score
            v = tl.load(V + k_pos * stride_v + hd).to(tl.float32)
            acc = acc * (l_prev * exp_prev / l_new) + v * (exp_score / l_new)
            m_prev = m_new
            l_prev = l_new
        tl.store(Out + q_pos * stride_o + hd, acc)

    ln_sig = {'X': '*fp32', 'Y': '*fp32', 'W': '*fp32', 'B': '*fp32',
              'Mean': '*fp32', 'Rstd': '*fp32', 'stride': 'i32', 'N': 'i32',
              'eps': 'fp32', 'BLOCK_SIZE': 'constexpr'}
    lin_sig = {'X': '*fp32', 'W': '*fp32', 'Bias': '*fp32', 'Y': '*fp32',
               'stride_x': 'i32', 'stride_w': 'i32', 'N_out': 'i32',
               'BLOCK_K': 'constexpr'}
    attn_sig = {'Q': '*fp32', 'K': '*fp32', 'V': '*fp32', 'Out': '*fp32',
                'stride_q': 'i32', 'stride_k': 'i32', 'stride_v': 'i32',
                'stride_o': 'i32', 'seq_len': 'i32', 'scale': 'fp32',
                'BLOCK_HD': 'constexpr'}

    # Helper: compile and run via compile_and_run
    def run_ln(x_arr, w, b):
        T, E = x_arr.shape
        return compile_and_run(
            ln_k, ln_sig, {'BLOCK_SIZE': BS}, grid=(T,),
            buffers={'X': x_arr.ravel(), 'Y': np.zeros(T * E, np.float32),
                     'W': w, 'B': b,
                     'Mean': np.zeros(T, np.float32),
                     'Rstd': np.zeros(T, np.float32)},
            scalars={'stride': E, 'N': E, 'eps': 1e-5},
            num_warps=nw,
        )['Y'].reshape(T, E)

    def run_linear(x_arr, w_arr, bias, out_n, bk, nw_lin):
        T, K = x_arr.shape
        if K < bk:
            xp = np.zeros((T, bk), np.float32); xp[:, :K] = x_arr
            wp = np.zeros((out_n, bk), np.float32); wp[:, :K] = w_arr
        else:
            xp, wp = x_arr, w_arr
        return compile_and_run(
            lin_k, lin_sig, {'BLOCK_K': bk}, grid=(T, out_n),
            buffers={'X': xp.ravel(), 'W': wp.ravel(), 'Bias': bias,
                     'Y': np.zeros(T * out_n, np.float32)},
            scalars={'stride_x': bk, 'stride_w': bk, 'N_out': out_n},
            num_warps=nw_lin,
        )['Y'].reshape(T, out_n)

    def run_attn(q, k, v):
        T, HD = q.shape
        BHD = _next_pow2(HD)
        nw_a = max(1, BHD // 32)
        if HD < BHD:
            qp = np.zeros((T, BHD), np.float32); qp[:, :HD] = q
            kp = np.zeros((T, BHD), np.float32); kp[:, :HD] = k
            vp = np.zeros((T, BHD), np.float32); vp[:, :HD] = v
        else:
            qp, kp, vp = q, k, v
        sc = float(1.0 / np.sqrt(HD))
        out = compile_and_run(
            attn_k, attn_sig, {'BLOCK_HD': BHD}, grid=(T,),
            buffers={'Q': qp.ravel(), 'K': kp.ravel(), 'V': vp.ravel(),
                     'Out': np.zeros(T * BHD, np.float32)},
            scalars={'stride_q': BHD, 'stride_k': BHD, 'stride_v': BHD,
                     'stride_o': BHD, 'seq_len': T, 'scale': sc},
            num_warps=nw_a,
        )['Out'].reshape(T, BHD)[:, :HD]
        return out

    # GPU forward pass
    token_ids = np.array([1, 42, 100, 200], dtype=np.int32)
    T = len(token_ids)
    x = weights["wte"][token_ids] + weights["wpe"][:T]

    for layer in range(n_layer):
        pfx = f"l{layer}_"
        ln1 = run_ln(x, weights[pfx + "ln1_w"], weights[pfx + "ln1_b"])
        qkv = run_linear(ln1, weights[pfx + "qkv_w"], weights[pfx + "qkv_b"],
                          3 * n_embd, BS, nw)
        Q = qkv[:, :n_embd].reshape(T, n_head, head_dim)
        K_h = qkv[:, n_embd:2*n_embd].reshape(T, n_head, head_dim)
        V_h = qkv[:, 2*n_embd:].reshape(T, n_head, head_dim)
        attn_out = np.zeros((T, n_head, head_dim), np.float32)
        for h in range(n_head):
            attn_out[:, h, :] = run_attn(Q[:, h].copy(), K_h[:, h].copy(), V_h[:, h].copy())
        proj = run_linear(attn_out.reshape(T, n_embd),
                           weights[pfx + "proj_w"], weights[pfx + "proj_b"],
                           n_embd, BS, nw)
        x = x + proj
        ln2 = run_ln(x, weights[pfx + "ln2_w"], weights[pfx + "ln2_b"])
        fc = run_linear(ln2, weights[pfx + "fc_w"], weights[pfx + "fc_b"],
                         4 * n_embd, BS, nw)
        # GELU on CPU (trivial)
        fc = (0.5 * fc * (1 + np.tanh(0.7978845608 * (fc + 0.044715 * fc**3)))).astype(np.float32)
        fc2 = run_linear(fc, weights[pfx + "fc2_w"], weights[pfx + "fc2_b"],
                           n_embd, BS4, nw4)
        x = x + fc2

    x = run_ln(x, weights["ln_f_w"], weights["ln_f_b"])
    logits = run_linear(x, weights["wte"], np.zeros(n_vocab, np.float32),
                         n_vocab, BS, nw)

    # NumPy reference
    x_ref = weights["wte"][token_ids] + weights["wpe"][:T]
    for layer in range(n_layer):
        pfx = f"l{layer}_"
        m = x_ref.mean(1, keepdims=True); v = x_ref.var(1, keepdims=True)
        ln1 = (x_ref - m) / np.sqrt(v + 1e-5) * weights[pfx + "ln1_w"] + weights[pfx + "ln1_b"]
        qkv = ln1 @ weights[pfx + "qkv_w"].T + weights[pfx + "qkv_b"]
        Q = qkv[:, :n_embd].reshape(T, n_head, head_dim)
        K_h = qkv[:, n_embd:2*n_embd].reshape(T, n_head, head_dim)
        V_h = qkv[:, 2*n_embd:].reshape(T, n_head, head_dim)
        sc = 1.0 / np.sqrt(head_dim)
        ao = np.zeros_like(Q)
        for h in range(n_head):
            s = Q[:, h] @ K_h[:, h].T * sc
            msk = np.triu(np.ones((T, T), bool), 1); s[msk] = -1e9
            e = np.exp(s - s.max(1, keepdims=True))
            ao[:, h] = (e / e.sum(1, keepdims=True)) @ V_h[:, h]
        p = ao.reshape(T, n_embd) @ weights[pfx + "proj_w"].T + weights[pfx + "proj_b"]
        x_ref = x_ref + p
        m = x_ref.mean(1, keepdims=True); v = x_ref.var(1, keepdims=True)
        ln2 = (x_ref - m) / np.sqrt(v + 1e-5) * weights[pfx + "ln2_w"] + weights[pfx + "ln2_b"]
        fc = ln2 @ weights[pfx + "fc_w"].T + weights[pfx + "fc_b"]
        fc = 0.5 * fc * (1 + np.tanh(0.7978845608 * (fc + 0.044715 * fc**3)))
        fc2 = fc @ weights[pfx + "fc2_w"].T + weights[pfx + "fc2_b"]
        x_ref = x_ref + fc2
    m = x_ref.mean(1, keepdims=True); v = x_ref.var(1, keepdims=True)
    x_ref = (x_ref - m) / np.sqrt(v + 1e-5) * weights["ln_f_w"] + weights["ln_f_b"]
    logits_ref = x_ref @ weights["wte"].T

    max_diff = np.abs(logits - logits_ref).max()
    assert max_diff < 0.01, f"GPT-2 pipeline max_diff={max_diff:.6f}"
    assert np.array_equal(logits.argmax(1), logits_ref.argmax(1)), \
        f"Argmax mismatch: gpu={logits.argmax(1)} ref={logits_ref.argmax(1)}"


# ============================================================================
# Loop-based Kernel Tests (for large N that exceeds single-pass workgroup size)
# ============================================================================

@triton.jit
def _ln_loop_kernel(X, Y, W, B, Mean, Rstd, stride, N, eps,
                    BLOCK: tl.constexpr):
    """Loop-based LayerNorm for arbitrary N."""
    row = tl.program_id(0)
    num_chunks = (N + BLOCK - 1) // BLOCK
    _sum = tl.zeros([1], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += tl.sum(x, axis=0)
    mean = tl.sum(_sum, axis=0) / N
    _var = tl.zeros([1], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        diff = tl.where(mask, diff, 0.0)
        _var += tl.sum(diff * diff, axis=0)
    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + row * stride + cols, y, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


@triton.jit
def _linear_loop_kernel(X, W, Bias, Y, K, stride_x, stride_w, N,
                        BLOCK_K: tl.constexpr):
    """Loop-based linear projection for arbitrary K."""
    row = tl.program_id(0)
    col = tl.program_id(1)
    num_chunks = (K + BLOCK_K - 1) // BLOCK_K
    acc = tl.zeros([1], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK_K
        ks = off + tl.arange(0, BLOCK_K)
        mask = ks < K
        x = tl.load(X + row * stride_x + ks, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + col * stride_w + ks, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * w, axis=0)
    dot = tl.sum(acc, axis=0)
    b = tl.load(Bias + col).to(tl.float32)
    tl.store(Y + row * N + col, dot + b)


def _run_loop_ln(T, N, BLOCK=128):
    """Run loop-based LayerNorm and return (gpu_out, ref_out)."""
    np.random.seed(42)
    x = np.random.randn(T, N).astype(np.float32) * 0.5
    w = np.random.randn(N).astype(np.float32) * 0.1 + 1.0
    b = np.random.randn(N).astype(np.float32) * 0.01

    nw = max(1, BLOCK // 32)
    sig = {'X': '*fp32', 'Y': '*fp32', 'W': '*fp32', 'B': '*fp32',
           'Mean': '*fp32', 'Rstd': '*fp32', 'stride': 'i32', 'N': 'i32',
           'eps': 'fp32', 'BLOCK': 'constexpr'}
    out = compile_and_run(
        _ln_loop_kernel, sig, {'BLOCK': BLOCK}, grid=(T,),
        buffers={'X': x.ravel(), 'Y': np.zeros(T * N, np.float32),
                 'W': w, 'B': b,
                 'Mean': np.zeros(T, np.float32),
                 'Rstd': np.zeros(T, np.float32)},
        scalars={'stride': N, 'N': N, 'eps': 1e-5},
        num_warps=nw)

    gpu = out['Y'].reshape(T, N)
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    ref = (x - mean) / np.sqrt(var + 1e-5) * w + b
    return gpu, ref


def _run_loop_linear(T, K, N, BLOCK_K=128):
    """Run loop-based linear projection and return (gpu_out, ref_out)."""
    np.random.seed(42)
    x = np.random.randn(T, K).astype(np.float32) * 0.1
    W = np.random.randn(N, K).astype(np.float32) * 0.02
    bias = np.random.randn(N).astype(np.float32) * 0.01

    nw = max(1, BLOCK_K // 32)
    sig = {'X': '*fp32', 'W': '*fp32', 'Bias': '*fp32', 'Y': '*fp32',
           'K': 'i32', 'stride_x': 'i32', 'stride_w': 'i32', 'N': 'i32',
           'BLOCK_K': 'constexpr'}
    out = compile_and_run(
        _linear_loop_kernel, sig, {'BLOCK_K': BLOCK_K}, grid=(T, N),
        buffers={'X': x.ravel(), 'W': W.ravel(), 'Bias': bias,
                 'Y': np.zeros(T * N, np.float32)},
        scalars={'K': K, 'stride_x': K, 'stride_w': K, 'N': N},
        num_warps=nw)

    gpu = out['Y'].reshape(T, N)
    ref = x @ W.T + bias
    return gpu, ref


def test_loop_ln_256():
    """Loop-based LayerNorm with N=256, BLOCK=128 (2 chunks)."""
    gpu, ref = _run_loop_ln(4, 256, BLOCK=128)
    assert_close(gpu, ref, atol=1e-4, name="loop_ln_256")


def test_loop_ln_768():
    """Loop-based LayerNorm with N=768, BLOCK=128 (6 chunks) — GPT-2 scale."""
    gpu, ref = _run_loop_ln(2, 768, BLOCK=128)
    assert_close(gpu, ref, atol=1e-4, name="loop_ln_768")


def test_loop_linear_768_to_2304():
    """Loop-based linear K=768->N=2304, BLOCK_K=128 — GPT-2 QKV projection."""
    gpu, ref = _run_loop_linear(2, 768, 2304, BLOCK_K=128)
    assert_close(gpu, ref, atol=1e-3, name="loop_linear_768_2304")


def test_loop_linear_3072_to_768():
    """Loop-based linear K=3072->N=768, BLOCK_K=128 — GPT-2 MLP down-project."""
    gpu, ref = _run_loop_linear(2, 3072, 768, BLOCK_K=128)
    assert_close(gpu, ref, atol=1e-3, name="loop_linear_3072_768")


def test_gpt2_pipeline_768():
    """Full GPT-2 pipeline at real dimensions: 768 embd, 12 heads, 2 layers."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__),
                                     '..', '..', '..', 'examples'))
    from webgpu_gpt2 import GPT2WebGPU

    n_layer, n_head, n_embd, n_vocab = 2, 12, 768, 256
    head_dim = n_embd // n_head
    np.random.seed(42)

    weights = {}
    weights["wte.weight"] = np.random.randn(n_vocab, n_embd).astype(np.float32) * 0.02
    weights["wpe.weight"] = np.random.randn(1024, n_embd).astype(np.float32) * 0.02
    weights["ln_f.weight"] = np.ones(n_embd, dtype=np.float32)
    weights["ln_f.bias"] = np.zeros(n_embd, dtype=np.float32)
    for i in range(n_layer):
        pfx = f"h.{i}."
        weights[pfx + "ln_1.weight"] = np.ones(n_embd, dtype=np.float32)
        weights[pfx + "ln_1.bias"] = np.zeros(n_embd, dtype=np.float32)
        weights[pfx + "ln_2.weight"] = np.ones(n_embd, dtype=np.float32)
        weights[pfx + "ln_2.bias"] = np.zeros(n_embd, dtype=np.float32)
        weights[pfx + "attn.c_attn.weight"] = np.random.randn(3*n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "attn.c_attn.bias"] = np.zeros(3*n_embd, dtype=np.float32)
        weights[pfx + "attn.c_proj.weight"] = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "attn.c_proj.bias"] = np.zeros(n_embd, dtype=np.float32)
        weights[pfx + "mlp.c_fc.weight"] = np.random.randn(4*n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.c_fc.bias"] = np.zeros(4*n_embd, dtype=np.float32)
        weights[pfx + "mlp.c_proj.weight"] = np.random.randn(n_embd, 4*n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.c_proj.bias"] = np.zeros(n_embd, dtype=np.float32)

    model = GPT2WebGPU(weights, n_layer=n_layer, n_head=n_head,
                       n_embd=n_embd, n_vocab=n_vocab)
    token_ids = np.array([1, 42, 100, 200], dtype=np.int32)
    logits = model.forward(token_ids)

    # NumPy reference
    T = len(token_ids)
    x = weights["wte.weight"][token_ids] + weights["wpe.weight"][:T]
    for layer in range(n_layer):
        pfx = f"h.{layer}."
        m = x.mean(1, keepdims=True); v = x.var(1, keepdims=True)
        ln1 = (x-m)/np.sqrt(v+1e-5)*weights[pfx+"ln_1.weight"]+weights[pfx+"ln_1.bias"]
        qkv = ln1@weights[pfx+"attn.c_attn.weight"].T+weights[pfx+"attn.c_attn.bias"]
        Q=qkv[:,:n_embd].reshape(T,n_head,head_dim)
        Kh=qkv[:,n_embd:2*n_embd].reshape(T,n_head,head_dim)
        Vh=qkv[:,2*n_embd:].reshape(T,n_head,head_dim)
        sc=1.0/np.sqrt(head_dim); ao=np.zeros_like(Q)
        for h in range(n_head):
            s=Q[:,h]@Kh[:,h].T*sc
            msk=np.triu(np.ones((T,T),bool),1); s[msk]=-1e9
            e=np.exp(s-s.max(1,keepdims=True))
            ao[:,h]=(e/e.sum(1,keepdims=True))@Vh[:,h]
        p=ao.reshape(T,n_embd)@weights[pfx+"attn.c_proj.weight"].T+weights[pfx+"attn.c_proj.bias"]
        x=x+p
        m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True)
        ln2=(x-m)/np.sqrt(v+1e-5)*weights[pfx+"ln_2.weight"]+weights[pfx+"ln_2.bias"]
        fc=ln2@weights[pfx+"mlp.c_fc.weight"].T+weights[pfx+"mlp.c_fc.bias"]
        fc=0.5*fc*(1+np.tanh(0.7978845608*(fc+0.044715*fc**3)))
        fc2=fc@weights[pfx+"mlp.c_proj.weight"].T+weights[pfx+"mlp.c_proj.bias"]
        x=x+fc2
    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True)
    x=(x-m)/np.sqrt(v+1e-5)*weights["ln_f.weight"]+weights["ln_f.bias"]
    logits_ref = x @ weights["wte.weight"].T

    max_diff = np.abs(logits - logits_ref).max()
    assert max_diff < 0.01, f"GPT-2 768 pipeline max_diff={max_diff:.6f}"
    assert np.array_equal(logits.argmax(1), logits_ref.argmax(1)), \
        f"Argmax mismatch: gpu={logits.argmax(1)} ref={logits_ref.argmax(1)}"


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
