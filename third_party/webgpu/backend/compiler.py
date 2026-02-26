"""
WebGPU Backend Compiler for Triton
===================================

Compilation pipeline: ttir -> ttgir -> llir -> spv

Follows the Intel XPU approach of generating SPIR-V from LLVM IR.
Dawn consumes the SPIR-V compute shaders directly.
"""

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, webgpu

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import re
import os
from pathlib import Path


@dataclass(frozen=True)
class WebGPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    warp_size: int = 32
    enable_fp_fusion: bool = True
    supported_fp8_dtypes: Tuple[str] = ()
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'webgpu'
    sanitize_overflow: bool = True
    arch: str = None

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class WebGPUBackend(BaseBackend):
    """
    WebGPU Backend for Triton.

    Generates SPIR-V compute shaders that run on Dawn's WebGPU implementation.
    Pipeline: Triton IR -> TritonGPU IR -> LLVM IR -> SPIR-V
    """

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'webgpu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "wgsl"

    def parse_options(self, opts) -> Any:
        args = {'arch': f"webgpu{self.target.arch}"}
        args.update({
            k: opts[k]
            for k in WebGPUOptions.__dataclass_fields__.keys()
            if k in opts and opts[k] is not None
        })
        return WebGPUOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        codegen_fns = {
            # FMA-based dot has no minimum shape constraints
            "min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1),
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def load_dialects(self, ctx):
        webgpu.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        """Optimize Triton IR (same as NVIDIA/Intel pipeline)."""
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt):
        """Convert Triton IR to TritonGPU IR for WebGPU target."""
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # Convert to TritonGPU IR targeting the WebGPU device
        passes.ttir.add_convert_to_ttgpuir(
            pm, "webgpu:0", opt.num_warps, opt.warp_size, opt.num_ctas
        )

        # Standard optimizations
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        # Decompose mixed-precision dots (e.g. f16 inputs → f32 accumulator)
        # and accelerate matmul patterns. The MMA acceleration patterns target
        # NVIDIA-specific encodings (no-op for WebGPU), but the mixed-mode
        # decomposition inserts arith.extf so convertFMADot sees uniform types.
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, False)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)

        pm.run(mod, 'make_ttgir')
        return mod

    def make_llir(self, src, metadata, options):
        """Convert TritonGPU IR to LLVM IR targeting SPIR-V."""
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        # Lower SCF (for/if/while) to CF (branch/cond_br) BEFORE the main
        # conversion pass, matching NVIDIA's pipeline ordering.  The main
        # conversion pass then handles CF ops together with TritonGPU ops
        # using the same TritonGPU type-converter, so tensor-typed block
        # args are consistently converted to LLVM struct types.
        passes.convert.add_scf_to_cf(pm)

        # Allocate shared memory
        passes.ttgpuir.add_allocate_shared_memory(pm)

        # WebGPU-specific TritonGPU → LLVM conversion pass
        # This lowers all TritonGPU ops to LLVM dialect using
        # generic conversion patterns + WebGPU-specific patterns.
        # CF→LLVM lowering is handled at the end of this pass.
        webgpu.passes.ttgpuir.add_to_llvmir(pm)

        passes.convert.add_arith_to_llvmir(pm)
        passes.convert.add_index_to_llvmir(pm)

        passes.ttgpuir.add_canonicalize_llvm_ir(pm)

        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        pm.run(mod, 'make_llir')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

        # Set SPIR-V target triple and data layout directly
        # (we can't use llvm.attach_datalayout because the bundled LLVM
        # doesn't include the SPIR-V target backend)
        webgpu.set_spv_target_triple(llvm_mod)

        # Link external libs if any
        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        # Skip target-specific optimizations (no SPIR-V target machine available)
        # llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Extract metadata
        metadata["shared"] = src.get_int_attr("ttg.shared") or 0
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size") or 0
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment") or 1

        ret = str(llvm_mod)
        del llvm_mod
        del context
        # Save LLVM IR for WGSL translation stage
        metadata["llir_str"] = ret

        # Extract kernel name from LLVM IR (first 'define' function)
        import re as _re
        m = _re.search(r'define\s+\S+\s+@(\w+)\s*\(', ret)
        metadata["name"] = m.group(1) if m else "kernel"

        return ret

    @staticmethod
    def make_spv(src, metadata, options):
        """Translate LLVM IR to SPIR-V binary."""
        spirv, name = webgpu.translate_to_spirv(src)
        metadata["name"] = name
        return spirv

    @staticmethod
    def make_wgsl(src, metadata, options):
        """Translate LLVM IR to WGSL compute shader for GPU execution via wgpu.

        This stage runs after SPV.  The LLVM IR was saved in metadata by
        make_llir, so we read it from there (src is the SPV binary, which
        we ignore).

        A ``signature`` entry must be present in *metadata* (added
        by WebGPUBackend.make_ttir or by caller) for full type
        information; otherwise types are inferred from the LLVM IR.
        """
        from .llvm_to_wgsl import translate_llvm_to_wgsl

        # Retrieve the LLVM IR saved by make_llir
        llir = metadata.get('llir_str', '')
        if not llir:
            raise ValueError("No LLVM IR found in metadata for WGSL translation")

        sig = metadata.get('signature', {})
        num_warps = options.num_warps if hasattr(options, 'num_warps') else 4
        warp_size = options.warp_size if hasattr(options, 'warp_size') else 32

        try:
            result = translate_llvm_to_wgsl(llir, sig, num_warps, warp_size)
            metadata['wgsl_bindings'] = result.buffer_bindings
            metadata['wgsl_params'] = result.param_fields
            metadata['wgsl_workgroup_size'] = result.workgroup_size
            return result.wgsl
        except Exception as e:
            # WGSL translation is best-effort; don't break compilation
            metadata['wgsl_error'] = str(e)
            return f"// WGSL translation failed: {e}\n"

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        # Skip SPIR-V — make_wgsl reads LLVM IR directly from metadata,
        # and Dawn compiles WGSL to the backend's native format (DXIL/SPIR-V).
        stages["wgsl"] = lambda src, metadata: self.make_wgsl(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return f'SPIRV-webgpu-{self.target.arch}'
