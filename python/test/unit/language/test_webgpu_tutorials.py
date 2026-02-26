"""
Compile-only tests for the WebGPU backend based on the official Triton tutorials.
Each test compiles the tutorial kernel through the full WebGPU pipeline
(ttir -> ttgir -> llir -> spv) without executing, since the WebGPU driver
(Dawn) is not yet integrated.

Tutorials 01-05 are fully tested (all kernels).
Tutorials 06-11 use NVIDIA/AMD-specific features (TMA descriptors, warp
specialization, libdevice, GDC, dot_scaled, etc.) and are tested via
simplified equivalents that exercise the same computational patterns on
WebGPU.
"""
import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

WEBGPU_TARGET = GPUTarget("webgpu", 0, 32)


def compile_webgpu(fn, signature, constexprs=None):
    """Helper to compile a kernel for WebGPU and return the compiled artifact."""
    if constexprs is None:
        constexprs = {}
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=WEBGPU_TARGET)


# ============================================================================
# Tutorial 01 — Vector Addition
# ============================================================================

class TestTutorial01VectorAdd:
    """From python/tutorials/01-vector-add.py"""

    def test_add_kernel(self):
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

        k = compile_webgpu(add_kernel, {
            "x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32",
            "n_elements": "i32", "BLOCK_SIZE": "constexpr"
        }, constexprs={"BLOCK_SIZE": 1024})
        assert "spv" in k.asm
        assert "ttir" in k.asm
        assert "ttgir" in k.asm
        assert "llir" in k.asm


# ============================================================================
# Tutorial 02 — Fused Softmax
# ============================================================================

class TestTutorial02FusedSoftmax:
    """From python/tutorials/02-fused-softmax.py"""

    def test_softmax_kernel(self):
        @triton.jit
        def softmax_kernel(output_ptr, input_ptr, input_row_stride,
                           output_row_stride, n_rows, n_cols,
                           BLOCK_SIZE: tl.constexpr,
                           num_stages: tl.constexpr):
            row_start = tl.program_id(0)
            row_step = tl.num_programs(0)
            for row_idx in tl.range(row_start, n_rows, row_step,
                                    num_stages=num_stages):
                row_start_ptr = input_ptr + row_idx * input_row_stride
                col_offsets = tl.arange(0, BLOCK_SIZE)
                input_ptrs = row_start_ptr + col_offsets
                mask = col_offsets < n_cols
                row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
                row_minus_max = row - tl.max(row, axis=0)
                numerator = tl.exp(row_minus_max)
                denominator = tl.sum(numerator, axis=0)
                softmax_output = numerator / denominator
                output_row_start_ptr = output_ptr + row_idx * output_row_stride
                output_ptrs = output_row_start_ptr + col_offsets
                tl.store(output_ptrs, softmax_output, mask=mask)

        k = compile_webgpu(softmax_kernel, {
            "output_ptr": "*fp32", "input_ptr": "*fp32",
            "input_row_stride": "i32", "output_row_stride": "i32",
            "n_rows": "i32", "n_cols": "i32",
            "BLOCK_SIZE": "constexpr", "num_stages": "constexpr"
        }, constexprs={"BLOCK_SIZE": 1024, "num_stages": 4})
        assert "spv" in k.asm


# ============================================================================
# Tutorial 03 — Matrix Multiplication
# ============================================================================

class TestTutorial03MatMul:
    """From python/tutorials/03-matrix-multiplication.py"""

    def test_matmul_kernel_no_activation(self):
        @triton.jit
        def leaky_relu(x):
            return tl.where(x >= 0, x, 0.01 * x)

        @triton.jit
        def matmul_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr,
            ACTIVATION: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                              offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                              offs_bn[None, :] * stride_bn)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                   dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(a_ptrs,
                            mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                            other=0.0)
                b = tl.load(b_ptrs,
                            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                            other=0.0)
                accumulator = tl.dot(a, b, accumulator)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
            if ACTIVATION == "leaky_relu":
                accumulator = leaky_relu(accumulator)
            c = accumulator.to(tl.float16)

            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + \
                stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)

        k = compile_webgpu(matmul_kernel, {
            "a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16",
            "M": "i32", "N": "i32", "K": "i32",
            "stride_am": "i32", "stride_ak": "i32",
            "stride_bk": "i32", "stride_bn": "i32",
            "stride_cm": "i32", "stride_cn": "i32",
            "BLOCK_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr",
            "BLOCK_SIZE_K": "constexpr",
            "GROUP_SIZE_M": "constexpr", "ACTIVATION": "constexpr",
        }, constexprs={
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8,
            "ACTIVATION": "",
        })
        assert "spv" in k.asm

    def test_matmul_kernel_with_leaky_relu(self):
        @triton.jit
        def leaky_relu(x):
            return tl.where(x >= 0, x, 0.01 * x)

        @triton.jit
        def matmul_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr,
            ACTIVATION: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                              offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                              offs_bn[None, :] * stride_bn)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                   dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(a_ptrs,
                            mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                            other=0.0)
                b = tl.load(b_ptrs,
                            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                            other=0.0)
                accumulator = tl.dot(a, b, accumulator)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
            if ACTIVATION == "leaky_relu":
                accumulator = leaky_relu(accumulator)
            c = accumulator.to(tl.float16)

            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + \
                stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)

        k = compile_webgpu(matmul_kernel, {
            "a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16",
            "M": "i32", "N": "i32", "K": "i32",
            "stride_am": "i32", "stride_ak": "i32",
            "stride_bk": "i32", "stride_bn": "i32",
            "stride_cm": "i32", "stride_cn": "i32",
            "BLOCK_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr",
            "BLOCK_SIZE_K": "constexpr",
            "GROUP_SIZE_M": "constexpr", "ACTIVATION": "constexpr",
        }, constexprs={
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8,
            "ACTIVATION": "leaky_relu",
        })
        assert "spv" in k.asm


# ============================================================================
# Tutorial 04 — Low-Memory Dropout
# ============================================================================

class TestTutorial04Dropout:
    """From python/tutorials/04-low-memory-dropout.py"""

    def test_dropout_kernel(self):
        @triton.jit
        def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p,
                     BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
            output = tl.where(x_keep, x / (1 - p), 0.0)
            tl.store(output_ptr + offsets, output, mask=mask)

        k = compile_webgpu(_dropout, {
            "x_ptr": "*fp32", "x_keep_ptr": "*i32", "output_ptr": "*fp32",
            "n_elements": "i32", "p": "fp32",
            "BLOCK_SIZE": "constexpr"
        }, constexprs={"BLOCK_SIZE": 1024})
        assert "spv" in k.asm

    def test_seeded_dropout_kernel(self):
        @triton.jit
        def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed,
                            BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            random = tl.rand(seed, offsets)
            x_keep = random > p
            output = tl.where(x_keep, x / (1 - p), 0.0)
            tl.store(output_ptr + offsets, output, mask=mask)

        k = compile_webgpu(_seeded_dropout, {
            "x_ptr": "*fp32", "output_ptr": "*fp32",
            "n_elements": "i32", "p": "fp32", "seed": "i32",
            "BLOCK_SIZE": "constexpr"
        }, constexprs={"BLOCK_SIZE": 1024})
        assert "spv" in k.asm


# ============================================================================
# Tutorial 05 — Layer Normalization
# ============================================================================

class TestTutorial05LayerNorm:
    """From python/tutorials/05-layer-norm.py"""

    def test_layer_norm_fwd_fused(self):
        @triton.jit
        def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride, N, eps,
                                  BLOCK_SIZE: tl.constexpr):
            row = tl.program_id(0)
            Y += row * stride
            X += row * stride
            # Compute mean
            _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
                _mean += a
            mean = tl.sum(_mean, axis=0) / N
            # Compute variance
            _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
                x = tl.where(cols < N, x - mean, 0.)
                _var += x * x
            var = tl.sum(_var, axis=0) / N
            rstd = 1 / tl.sqrt(var + eps)
            # Store mean/rstd
            tl.store(Mean + row, mean)
            tl.store(Rstd + row, rstd)
            # Normalize and apply linear transformation
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                w = tl.load(W + cols, mask=mask)
                b = tl.load(B + cols, mask=mask)
                x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
                x_hat = (x - mean) * rstd
                y = x_hat * w + b
                tl.store(Y + cols, y, mask=mask)

        k = compile_webgpu(_layer_norm_fwd_fused, {
            "X": "*fp32", "Y": "*fp32", "W": "*fp32", "B": "*fp32",
            "Mean": "*fp32", "Rstd": "*fp32",
            "stride": "i32", "N": "i32", "eps": "fp32",
            "BLOCK_SIZE": "constexpr"
        }, constexprs={"BLOCK_SIZE": 1024})
        assert "spv" in k.asm

    def test_layer_norm_bwd_dx_fused(self):
        @triton.jit
        def _layer_norm_bwd_dx_fused(DX, DY, DW, DB, X, W, Mean, Rstd,
                                     Lock, stride, N,
                                     GROUP_SIZE_M: tl.constexpr,
                                     BLOCK_SIZE_N: tl.constexpr):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK_SIZE_N)
            mask = cols < N
            X += row * stride
            DY += row * stride
            DX += row * stride
            # Load
            x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + cols, mask=mask).to(tl.float32)
            mean = tl.load(Mean + row)
            rstd = tl.load(Rstd + row)
            # Compute dx
            xhat = (x - mean) * rstd
            wdy = w * dy
            xhat_wdy = xhat * wdy
            c1 = tl.sum(xhat_wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
            tl.store(DX + cols, dx, mask=mask)
            # Partial sums for dw/db (lock-based accumulation)
            partial_dw = (dy * xhat).to(w.dtype)
            partial_db = dy.to(w.dtype)
            # Accumulate partial sums using lock
            lock_id = row % GROUP_SIZE_M
            Lock += lock_id
            Count = Lock + GROUP_SIZE_M
            while tl.atomic_cas(Lock, 0, 1) == 1:
                pass
            count = tl.load(Count)
            if count == 0:
                tl.store(DW + lock_id * N + cols, partial_dw, mask=mask)
                tl.store(DB + lock_id * N + cols, partial_db, mask=mask)
            else:
                partial_dw += tl.load(DW + lock_id * N + cols, mask=mask)
                partial_db += tl.load(DB + lock_id * N + cols, mask=mask)
                tl.store(DW + lock_id * N + cols, partial_dw, mask=mask)
                tl.store(DB + lock_id * N + cols, partial_db, mask=mask)
            tl.store(Count, count + 1)
            tl.atomic_xchg(Lock, 0)

        k = compile_webgpu(_layer_norm_bwd_dx_fused, {
            "DX": "*fp32", "DY": "*fp32", "DW": "*fp32", "DB": "*fp32",
            "X": "*fp32", "W": "*fp32", "Mean": "*fp32", "Rstd": "*fp32",
            "Lock": "*i32", "stride": "i32", "N": "i32",
            "GROUP_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr"
        }, constexprs={"GROUP_SIZE_M": 64, "BLOCK_SIZE_N": 1024})
        assert "spv" in k.asm

    def test_layer_norm_bwd_dwdb(self):
        @triton.jit
        def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr):
            pid = tl.program_id(0)
            cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for i in range(0, M, BLOCK_SIZE_M):
                rows = i + tl.arange(0, BLOCK_SIZE_M)
                mask = (rows[:, None] < M) & (cols[None, :] < N)
                offs = rows[:, None] * N + cols[None, :]
                dw += tl.load(DW + offs, mask=mask, other=0.)
                db += tl.load(DB + offs, mask=mask, other=0.)
            sum_dw = tl.sum(dw, axis=0)
            sum_db = tl.sum(db, axis=0)
            tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
            tl.store(FINAL_DB + cols, sum_db, mask=cols < N)

        k = compile_webgpu(_layer_norm_bwd_dwdb, {
            "DW": "*fp32", "DB": "*fp32",
            "FINAL_DW": "*fp32", "FINAL_DB": "*fp32",
            "M": "i32", "N": "i32",
            "BLOCK_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr"
        }, constexprs={"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128})
        assert "spv" in k.asm


# ============================================================================
# Tutorial 06 — Fused Attention (simplified; original uses TMA/Hopper)
# ============================================================================

class TestTutorial06FusedAttention:
    """Simplified Flash Attention pattern (dot + softmax + dot).
    The original tutorial uses Hopper TMA descriptors, warp specialization,
    exp2/log2, etc. which are NVIDIA-specific.
    """

    def test_attention_forward_simplified(self):
        @triton.jit
        def attention_fwd_kernel(
            Q, K, V, Out,
            stride_qm, stride_qd,
            stride_km, stride_kd,
            stride_vm, stride_vd,
            stride_om, stride_od,
            N_CTX, HEAD_DIM: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, HEAD_DIM)
            # Load Q block
            q_ptrs = Q + offs_m[:, None] * stride_qm + \
                offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
            # Initialize accumulators
            m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            # Loop over K/V blocks
            for start_n in range(0, N_CTX, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                # Load K block
                k_ptrs = K + offs_n[:, None] * stride_km + \
                    offs_d[None, :] * stride_kd
                k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
                # QK^T
                qk = tl.dot(q, tl.trans(k))
                # Online softmax update
                m_ij = tl.max(qk, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                beta = tl.exp(m_ij - m_new)
                l_i = l_i * alpha + tl.sum(tl.exp(qk - m_new[:, None]),
                                           axis=1)
                acc = acc * alpha[:, None]
                # Load V and accumulate
                v_ptrs = V + offs_n[:, None] * stride_vm + \
                    offs_d[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
                p = tl.exp(qk - m_new[:, None])
                acc += tl.dot(p.to(tl.float16), v)
                m_i = m_new
            # Finalize
            acc = acc / l_i[:, None]
            out_ptrs = Out + offs_m[:, None] * stride_om + \
                offs_d[None, :] * stride_od
            tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)

        k = compile_webgpu(attention_fwd_kernel, {
            "Q": "*fp16", "K": "*fp16", "V": "*fp16", "Out": "*fp32",
            "stride_qm": "i32", "stride_qd": "i32",
            "stride_km": "i32", "stride_kd": "i32",
            "stride_vm": "i32", "stride_vd": "i32",
            "stride_om": "i32", "stride_od": "i32",
            "N_CTX": "i32",
            "HEAD_DIM": "constexpr",
            "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
        }, constexprs={"HEAD_DIM": 32, "BLOCK_M": 32, "BLOCK_N": 32})
        assert "spv" in k.asm


# ============================================================================
# Tutorial 07 — Extern Functions (simplified; original uses libdevice)
# ============================================================================

class TestTutorial07ExternFunctions:
    """The original uses libdevice.asin (CUDA-specific extern).
    We test the same elementwise-math-over-block pattern using built-in
    tl.math functions that the WebGPU backend supports.
    """

    def test_math_function_kernel(self):
        @triton.jit
        def math_kernel(x_ptr, y_ptr, n_elements,
                        BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            # Use sin instead of libdevice.asin — same pattern
            y = tl.sin(x)
            tl.store(y_ptr + offsets, y, mask=mask)

        k = compile_webgpu(math_kernel, {
            "x_ptr": "*fp32", "y_ptr": "*fp32", "n_elements": "i32",
            "BLOCK_SIZE": "constexpr"
        }, constexprs={"BLOCK_SIZE": 1024})
        assert "spv" in k.asm


# ============================================================================
# Tutorial 08 — Grouped GEMM (simplified; original uses TMA descriptors)
# ============================================================================

class TestTutorial08GroupedGemm:
    """Simplified grouped GEMM: loops over multiple small matmuls.
    The original uses device-side TMA descriptors which are Hopper-specific.
    """

    def test_grouped_matmul_kernel(self):
        @triton.jit
        def grouped_matmul_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            num_groups,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
        ):
            pid = tl.program_id(0)
            pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
            pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            a_ptrs = a_ptr + offs_am[:, None] * stride_am + \
                offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + \
                offs_bn[None, :] * stride_bn

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(a_ptrs,
                            mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                            other=0.0)
                b = tl.load(b_ptrs,
                            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                            other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + \
                offs_cn[None, :] * stride_cn
            mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, acc, mask=mask)

        k = compile_webgpu(grouped_matmul_kernel, {
            "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
            "M": "i32", "N": "i32", "K": "i32",
            "stride_am": "i32", "stride_ak": "i32",
            "stride_bk": "i32", "stride_bn": "i32",
            "stride_cm": "i32", "stride_cn": "i32",
            "num_groups": "i32",
            "BLOCK_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr",
            "BLOCK_SIZE_K": "constexpr",
        }, constexprs={
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32
        })
        assert "spv" in k.asm


# ============================================================================
# Tutorial 09 — Persistent Matmul (simplified; original uses TMA + warp spec)
# ============================================================================

class TestTutorial09PersistentMatmul:
    """Simplified persistent kernel: tile-scheduling loop over the full grid.
    The original uses TMA descriptors, host-side TensorDescriptors, warp
    specialization, and proton profiling — all NVIDIA-specific.
    """

    def test_persistent_matmul_kernel(self):
        @triton.jit
        def matmul_persistent_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
        ):
            pid = tl.program_id(0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles = num_pid_m * num_pid_n

            # Persistent: each SM loops over multiple tiles
            for tile_id in range(pid, num_tiles, NUM_SMS):
                num_pid_in_group = GROUP_SIZE_M * num_pid_n
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (tile_id % num_pid_in_group) % \
                    group_size_m
                pid_n = (tile_id % num_pid_in_group) // group_size_m

                offs_am = (pid_m * BLOCK_SIZE_M +
                           tl.arange(0, BLOCK_SIZE_M)) % M
                offs_bn = (pid_n * BLOCK_SIZE_N +
                           tl.arange(0, BLOCK_SIZE_N)) % N
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                a_ptrs = a_ptr + offs_am[:, None] * stride_am + \
                    offs_k[None, :] * stride_ak
                b_ptrs = b_ptr + offs_k[:, None] * stride_bk + \
                    offs_bn[None, :] * stride_bn

                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for ki in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    a = tl.load(a_ptrs,
                                mask=offs_k[None, :] < K - ki * BLOCK_SIZE_K,
                                other=0.0)
                    b = tl.load(b_ptrs,
                                mask=offs_k[:, None] < K - ki * BLOCK_SIZE_K,
                                other=0.0)
                    acc = tl.dot(a, b, acc)
                    a_ptrs += BLOCK_SIZE_K * stride_ak
                    b_ptrs += BLOCK_SIZE_K * stride_bk

                c = acc.to(tl.float16)
                offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + \
                    offs_cn[None, :] * stride_cn
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                tl.store(c_ptrs, c, mask=c_mask)

        k = compile_webgpu(matmul_persistent_kernel, {
            "a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16",
            "M": "i32", "N": "i32", "K": "i32",
            "stride_am": "i32", "stride_ak": "i32",
            "stride_bk": "i32", "stride_bn": "i32",
            "stride_cm": "i32", "stride_cn": "i32",
            "BLOCK_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr",
            "BLOCK_SIZE_K": "constexpr",
            "GROUP_SIZE_M": "constexpr", "NUM_SMS": "constexpr",
        }, constexprs={
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8, "NUM_SMS": 64,
        })
        assert "spv" in k.asm


# ============================================================================
# Tutorial 10 — Block-Scaled Matmul (simplified; original uses dot_scaled/FP4)
# ============================================================================

class TestTutorial10BlockScaledMatmul:
    """Simplified block-scaled matmul: regular matmul with per-block scale
    factors applied. The original uses tl.dot_scaled with FP4/FP8 types
    and 5D TMA descriptors — all NVIDIA/AMD-specific.
    """

    def test_block_scaled_matmul_kernel(self):
        @triton.jit
        def block_scaled_matmul_kernel(
            a_ptr, b_ptr, c_ptr,
            a_scale_ptr, b_scale_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            a_ptrs = a_ptr + offs_m[:, None] * stride_am + \
                offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + \
                offs_n[None, :] * stride_bn

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for ki in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs,
                            mask=offs_k[None, :] < K - ki * BLOCK_K,
                            other=0.0)
                b = tl.load(b_ptrs,
                            mask=offs_k[:, None] < K - ki * BLOCK_K,
                            other=0.0)
                # Load per-block scales
                a_scale = tl.load(a_scale_ptr + pid_m * tl.cdiv(K, BLOCK_K) +
                                  ki)
                b_scale = tl.load(b_scale_ptr + pid_n * tl.cdiv(K, BLOCK_K) +
                                  ki)
                acc += tl.dot(a, b) * a_scale * b_scale
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk

            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + \
                offs_cn[None, :] * stride_cn
            mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, acc, mask=mask)

        k = compile_webgpu(block_scaled_matmul_kernel, {
            "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
            "a_scale_ptr": "*fp32", "b_scale_ptr": "*fp32",
            "M": "i32", "N": "i32", "K": "i32",
            "stride_am": "i32", "stride_ak": "i32",
            "stride_bk": "i32", "stride_bn": "i32",
            "stride_cm": "i32", "stride_cn": "i32",
            "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
        }, constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32})
        assert "spv" in k.asm


# ============================================================================
# Tutorial 11 — Programmatic Dependent Launch (simplified; original uses GDC)
# ============================================================================

class TestTutorial11ProgrammaticDependentLaunch:
    """The original uses tl.extra.cuda.gdc_wait/gdc_launch_dependents which
    are CUDA compute-capability 9+ only. We test the same vector-add
    pattern without those intrinsics.
    """

    def test_add_kernel_without_gdc(self):
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(output_ptr + offsets, x + y, mask=mask)

        k = compile_webgpu(add_kernel, {
            "x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32",
            "n_elements": "i32", "BLOCK_SIZE": "constexpr"
        }, constexprs={"BLOCK_SIZE": 1024})
        assert "spv" in k.asm
