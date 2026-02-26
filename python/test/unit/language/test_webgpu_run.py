"""
WebGPU Runtime Correctness Tests
==================================

These tests compile Triton kernels for the WebGPU target, translate
the LLVM IR to WGSL, execute the compute shader on the GPU via Dawn
(Google's native WebGPU implementation), and verify the results
against numpy reference implementations.

Dawn supports D3D12 (Windows), Vulkan (Linux), and Metal (macOS)
natively, consuming WGSL shaders directly through its Tint compiler.

Requirements:
    Dawn built from source (webgpu_dawn.dll / libwebgpu_dawn.so)
    numpy

Run:
    pytest python/test/unit/language/test_webgpu_run.py -v
"""
import pytest
import numpy as np
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

# Import WebGPU runtime components
from triton.backends.webgpu.llvm_to_wgsl import translate_llvm_to_wgsl
from triton.backends.webgpu.dawn_runner import DawnRunner, HAS_DAWN

# Skip entire module if Dawn is not available
pytestmark = pytest.mark.skipif(not HAS_DAWN(), reason="Dawn WebGPU library not available")

WEBGPU_TARGET = GPUTarget("webgpu", 0, 32)

# Shared runner instance (reuse GPU device across tests)
_runner = None


def get_runner():
    global _runner
    if _runner is None:
        _runner = DawnRunner()
    return _runner


def compile_webgpu(fn, signature, constexprs=None):
    """Compile a Triton kernel for WebGPU and return the compiled artifact."""
    if constexprs is None:
        constexprs = {}
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=WEBGPU_TARGET)


def compile_and_run(fn, signature, constexprs, grid, buffers, scalars=None,
                    num_warps=4, warp_size=32):
    """
    Compile a Triton kernel for WebGPU, translate to WGSL, and execute on GPU.

    Args:
        fn: Triton JIT kernel function
        signature: Dict of parameter names to types (e.g. {'a': '*fp32', 'n': 'i32'})
        constexprs: Dict of compile-time constants (e.g. {'BLOCK_SIZE': 256})
        grid: Tuple of workgroup counts
        buffers: Dict of buffer_name → numpy array
        scalars: Dict of scalar_name → value
        num_warps: Warps per workgroup
        warp_size: Threads per warp

    Returns:
        Dict of output buffer names → numpy result arrays
    """
    scalars = scalars or {}

    # Strip constexprs from signature for the translator
    sig_no_constexpr = {k: v for k, v in signature.items() if v != 'constexpr'}

    # Compile
    k = compile_webgpu(fn, signature, constexprs)
    llir = k.asm['llir']

    # Translate LLVM IR → WGSL
    result = translate_llvm_to_wgsl(llir, sig_no_constexpr, num_warps, warp_size)

    # Execute on GPU
    runner = get_runner()
    outputs = runner.run_kernel(
        wgsl_code=result.wgsl,
        buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields,
        workgroup_size=result.workgroup_size,
        grid=grid,
        buffers=buffers,
        scalars=scalars,
    )
    return outputs


# ============================================================================
# Helper assertion functions
# ============================================================================

def assert_close(actual, expected, rtol=1e-5, atol=1e-5, name="output"):
    """Assert numpy arrays are close with a descriptive message."""
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol,
        err_msg=f"{name}: max_diff={np.max(np.abs(actual - expected)):.6e}"
    )


# ============================================================================
# Tutorial 01: Vector Addition
# ============================================================================

class TestVectorAdd:
    """Test vector add kernel (Tutorial 01) — the simplest end-to-end test."""

    BLOCK_SIZE = 256

    @staticmethod
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def _run_add(self, n, dtype=np.float32):
        """Run add kernel with given size and return (actual, expected)."""
        x = np.random.randn(n).astype(dtype)
        y = np.random.randn(n).astype(dtype)
        output = np.zeros(n, dtype=dtype)
        expected = x + y

        BLOCK_SIZE = self.BLOCK_SIZE
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        results = compile_and_run(
            fn=self.add_kernel,
            signature={
                'x_ptr': '*fp32', 'y_ptr': '*fp32', 'output_ptr': '*fp32',
                'n_elements': 'i32', 'BLOCK_SIZE': 'constexpr',
            },
            constexprs={'BLOCK_SIZE': BLOCK_SIZE},
            grid=grid,
            buffers={'x_ptr': x, 'y_ptr': y, 'output_ptr': output},
            scalars={'n_elements': n},
        )

        actual = results['output_ptr'][:n]
        return actual, expected

    def test_add_exact_block(self):
        """N is an exact multiple of BLOCK_SIZE."""
        actual, expected = self._run_add(256)
        assert_close(actual, expected, name="add_exact_block")

    def test_add_multiple_blocks(self):
        """N requires multiple workgroups."""
        actual, expected = self._run_add(1024)
        assert_close(actual, expected, name="add_multiple_blocks")

    def test_add_non_aligned(self):
        """N is not a multiple of BLOCK_SIZE (tests masking)."""
        actual, expected = self._run_add(1000)
        assert_close(actual, expected, name="add_non_aligned")

    def test_add_small(self):
        """N smaller than BLOCK_SIZE."""
        actual, expected = self._run_add(100)
        assert_close(actual, expected, name="add_small")

    def test_add_large(self):
        """Larger input to stress-test multiple workgroups."""
        actual, expected = self._run_add(8192)
        assert_close(actual, expected, name="add_large")

    def test_add_single_element(self):
        """Edge case: single element."""
        actual, expected = self._run_add(1)
        assert_close(actual, expected, name="add_single")


# ============================================================================
# Elementwise operations
# ============================================================================

class TestElementwise:
    """Test various elementwise operations."""

    BLOCK_SIZE = 256

    def _run_elementwise(self, kernel_fn, signature, constexprs, n,
                         input_buffers, scalars, output_name, expected):
        grid = ((n + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE,)
        results = compile_and_run(
            fn=kernel_fn,
            signature=signature,
            constexprs=constexprs,
            grid=grid,
            buffers=input_buffers,
            scalars=scalars,
        )
        actual = results[output_name][:n]
        return actual

    def test_multiply(self):
        """Element-wise multiplication."""
        @triton.jit
        def mul_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            a = tl.load(a_ptr + offs, mask=mask)
            b = tl.load(b_ptr + offs, mask=mask)
            tl.store(c_ptr + offs, a * b, mask=mask)

        n = 1024
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.zeros(n, dtype=np.float32)
        expected = a * b

        actual = self._run_elementwise(
            mul_kernel,
            {'a_ptr': '*fp32', 'b_ptr': '*fp32', 'c_ptr': '*fp32',
             'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
            {'BLOCK_SIZE': self.BLOCK_SIZE},
            n,
            {'a_ptr': a, 'b_ptr': b, 'c_ptr': c},
            {'n': n},
            'c_ptr', expected,
        )
        assert_close(actual, expected, name="multiply")

    def test_subtract(self):
        """Element-wise subtraction."""
        @triton.jit
        def sub_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            a = tl.load(a_ptr + offs, mask=mask)
            b = tl.load(b_ptr + offs, mask=mask)
            tl.store(c_ptr + offs, a - b, mask=mask)

        n = 1024
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.zeros(n, dtype=np.float32)
        expected = a - b

        actual = self._run_elementwise(
            sub_kernel,
            {'a_ptr': '*fp32', 'b_ptr': '*fp32', 'c_ptr': '*fp32',
             'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
            {'BLOCK_SIZE': self.BLOCK_SIZE},
            n,
            {'a_ptr': a, 'b_ptr': b, 'c_ptr': c},
            {'n': n},
            'c_ptr', expected,
        )
        assert_close(actual, expected, name="subtract")

    def test_scalar_multiply(self):
        """Multiply array by scalar constant."""
        @triton.jit
        def scale_kernel(a_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            a = tl.load(a_ptr + offs, mask=mask)
            tl.store(c_ptr + offs, a * 2.0, mask=mask)

        n = 1024
        a = np.random.randn(n).astype(np.float32)
        c = np.zeros(n, dtype=np.float32)
        expected = a * 2.0

        actual = self._run_elementwise(
            scale_kernel,
            {'a_ptr': '*fp32', 'c_ptr': '*fp32',
             'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
            {'BLOCK_SIZE': self.BLOCK_SIZE},
            n,
            {'a_ptr': a, 'c_ptr': c},
            {'n': n},
            'c_ptr', expected,
        )
        assert_close(actual, expected, name="scalar_multiply")

    def test_fused_multiply_add(self):
        """Fused multiply-add: c = a * b + d."""
        @triton.jit
        def fma_kernel(a_ptr, b_ptr, d_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            a = tl.load(a_ptr + offs, mask=mask)
            b = tl.load(b_ptr + offs, mask=mask)
            d = tl.load(d_ptr + offs, mask=mask)
            tl.store(c_ptr + offs, a * b + d, mask=mask)

        n = 1024
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        d = np.random.randn(n).astype(np.float32)
        c = np.zeros(n, dtype=np.float32)
        expected = a * b + d

        actual = self._run_elementwise(
            fma_kernel,
            {'a_ptr': '*fp32', 'b_ptr': '*fp32', 'd_ptr': '*fp32',
             'c_ptr': '*fp32', 'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
            {'BLOCK_SIZE': self.BLOCK_SIZE},
            n,
            {'a_ptr': a, 'b_ptr': b, 'd_ptr': d, 'c_ptr': c},
            {'n': n},
            'c_ptr', expected,
        )
        assert_close(actual, expected, name="fused_multiply_add")


# ============================================================================
# Integer operations
# ============================================================================

class TestIntegerOps:
    """Test integer arithmetic and bitwise operations."""

    BLOCK_SIZE = 256

    def test_int_add(self):
        """Integer addition."""
        @triton.jit
        def kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            a = tl.load(a_ptr + offs, mask=mask)
            b = tl.load(b_ptr + offs, mask=mask)
            tl.store(c_ptr + offs, a + b, mask=mask)

        n = 1024
        a = np.random.randint(-1000, 1000, n).astype(np.int32)
        b = np.random.randint(-1000, 1000, n).astype(np.int32)
        c = np.zeros(n, dtype=np.int32)
        expected = a + b

        grid = ((n + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE,)
        results = compile_and_run(
            fn=kernel,
            signature={'a_ptr': '*i32', 'b_ptr': '*i32', 'c_ptr': '*i32',
                       'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
            constexprs={'BLOCK_SIZE': self.BLOCK_SIZE},
            grid=grid,
            buffers={'a_ptr': a, 'b_ptr': b, 'c_ptr': c},
            scalars={'n': n},
        )
        actual = results['c_ptr'][:n]
        assert_close(actual.astype(float), expected.astype(float), atol=0, name="int_add")


# ============================================================================
# Memory access patterns
# ============================================================================

class TestMemoryAccess:
    """Test various memory access patterns."""

    def test_copy(self):
        """Simple memory copy (load then store)."""
        @triton.jit
        def copy_kernel(src_ptr, dst_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            val = tl.load(src_ptr + offs, mask=mask)
            tl.store(dst_ptr + offs, val, mask=mask)

        n = 2048
        BLOCK_SIZE = 256
        src = np.random.randn(n).astype(np.float32)
        dst = np.zeros(n, dtype=np.float32)

        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        results = compile_and_run(
            fn=copy_kernel,
            signature={'src_ptr': '*fp32', 'dst_ptr': '*fp32',
                       'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
            constexprs={'BLOCK_SIZE': BLOCK_SIZE},
            grid=grid,
            buffers={'src_ptr': src, 'dst_ptr': dst},
            scalars={'n': n},
        )
        actual = results['dst_ptr'][:n]
        assert_close(actual, src, atol=0, name="copy")

    def test_stride_access(self):
        """Access with non-unit stride (every other element)."""
        @triton.jit
        def stride_kernel(src_ptr, dst_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            # Read from source with stride 2
            src_offs = offs * 2
            src_mask = src_offs < (n * 2)
            val = tl.load(src_ptr + src_offs, mask=src_mask)
            tl.store(dst_ptr + offs, val, mask=mask)

        n = 512
        BLOCK_SIZE = 256
        src = np.random.randn(n * 2).astype(np.float32)
        dst = np.zeros(n, dtype=np.float32)
        expected = src[::2][:n]

        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        results = compile_and_run(
            fn=stride_kernel,
            signature={'src_ptr': '*fp32', 'dst_ptr': '*fp32',
                       'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
            constexprs={'BLOCK_SIZE': BLOCK_SIZE},
            grid=grid,
            buffers={'src_ptr': src, 'dst_ptr': dst},
            scalars={'n': n},
        )
        actual = results['dst_ptr'][:n]
        assert_close(actual, expected, atol=0, name="stride_access")


# ============================================================================
# WGSL Translation Tests (unit tests for the translator)
# ============================================================================

class TestWGSLTranslation:
    """Test that WGSL translation produces valid, parseable shader code."""

    def test_add_kernel_produces_wgsl(self):
        """Verify add kernel LLVM IR translates to valid WGSL."""
        @triton.jit
        def kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        k = compile_webgpu(
            kernel,
            {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32',
             'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
            {'BLOCK_SIZE': 256},
        )

        sig = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32', 'n': 'i32'}
        result = translate_llvm_to_wgsl(k.asm['llir'], sig)

        # Verify WGSL structure
        wgsl = result.wgsl
        assert '@compute' in wgsl
        assert '@workgroup_size' in wgsl
        assert 'fn main(' in wgsl
        assert 'workgroup_id' in wgsl
        assert 'local_invocation_id' in wgsl
        assert 'buf0' in wgsl  # x_ptr buffer
        assert 'buf1' in wgsl  # y_ptr buffer
        assert 'buf2' in wgsl  # out_ptr buffer
        assert 'Params' in wgsl  # scalar params struct

    def test_bindings_correct(self):
        """Verify buffer bindings are correctly generated."""
        @triton.jit
        def kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) +
                     tl.load(y_ptr + offs, mask=mask), mask=mask)

        k = compile_webgpu(
            kernel,
            {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32',
             'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
            {'BLOCK_SIZE': 256},
        )

        sig = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32', 'n': 'i32'}
        result = translate_llvm_to_wgsl(k.asm['llir'], sig)

        # Check buffer bindings
        ptr_bindings = [b for b in result.buffer_bindings if not b.name.startswith('_')]
        assert len(ptr_bindings) >= 3  # x, y, out

        # Check that out_ptr is read_write (it's stored to)
        out_binding = next(b for b in ptr_bindings if b.name == 'out_ptr')
        assert out_binding.access == 'read_write'

        # Check params
        assert len(result.param_fields) >= 1
        assert result.param_fields[0].name == 'n'
        assert result.param_fields[0].wgsl_type == 'i32'

    def test_wgsl_in_compiled_asm(self):
        """Verify WGSL stage is included in compiled kernel's asm dict."""
        @triton.jit
        def kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

        k = compile_webgpu(
            kernel,
            {'x_ptr': '*fp32', 'out_ptr': '*fp32',
             'n': 'i32', 'BLOCK_SIZE': 'constexpr'},
            {'BLOCK_SIZE': 256},
        )

        assert 'wgsl' in k.asm, "WGSL stage should be in compiled kernel asm"
        wgsl = k.asm['wgsl']
        assert '@compute' in wgsl


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformance:
    """Basic performance benchmarks. These print timing info but don't assert
    specific thresholds (hardware-dependent)."""

    def test_vector_add_bandwidth(self):
        """Measure effective bandwidth for vector addition."""
        import time

        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        N = 1 << 20  # 1M elements
        BLOCK_SIZE = 256
        x = np.random.randn(N).astype(np.float32)
        y = np.random.randn(N).astype(np.float32)
        out = np.zeros(N, dtype=np.float32)
        expected = x + y

        sig = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32',
               'n': 'i32', 'BLOCK_SIZE': 'constexpr'}
        sig_no_ce = {k: v for k, v in sig.items() if v != 'constexpr'}
        k = compile_webgpu(add_kernel, sig, {'BLOCK_SIZE': BLOCK_SIZE})
        llir = k.asm['llir']
        result = translate_llvm_to_wgsl(llir, sig_no_ce, 4, 32)

        runner = get_runner()
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        # Warmup
        runner.run_kernel(
            wgsl_code=result.wgsl,
            buffer_bindings=result.buffer_bindings,
            param_fields=result.param_fields,
            workgroup_size=result.workgroup_size,
            grid=grid,
            buffers={'x_ptr': x, 'y_ptr': y, 'out_ptr': out},
            scalars={'n': N},
        )

        # Timed run
        num_iters = 5
        t0 = time.perf_counter()
        for _ in range(num_iters):
            outputs = runner.run_kernel(
                wgsl_code=result.wgsl,
                buffer_bindings=result.buffer_bindings,
                param_fields=result.param_fields,
                workgroup_size=result.workgroup_size,
                grid=grid,
                buffers={'x_ptr': x, 'y_ptr': y, 'out_ptr': out},
                scalars={'n': N},
            )
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) / num_iters * 1000
        # 3 arrays × N × 4 bytes (2 reads + 1 write)
        bytes_transferred = 3 * N * 4
        bandwidth_gb_s = bytes_transferred / (elapsed_ms / 1000) / 1e9

        print(f"\n  Vector Add (N={N:,}): {elapsed_ms:.2f} ms, "
              f"{bandwidth_gb_s:.1f} GB/s effective bandwidth")

        # Verify correctness
        actual = outputs['out_ptr'][:N]
        assert_close(actual, expected, name="perf_add")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
