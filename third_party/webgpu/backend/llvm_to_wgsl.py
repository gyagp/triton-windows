"""
LLVM IR → WGSL Translator for Triton WebGPU Backend
=====================================================

Translates the LLVM IR produced by Triton's WebGPU compilation pipeline
into WGSL (WebGPU Shading Language) compute shaders that can be executed
via wgpu-py on any GPU supporting Vulkan, Metal, or DX12.

The LLVM IR from Triton follows predictable patterns:
  - Single function with SPIR-V builtins for thread/block IDs
  - addrspace(1) pointers for global memory (storage buffers)
  - addrspace(3) for shared memory (workgroup storage)
  - Standard arithmetic, bitwise, comparison, and memory ops

This translator handles straight-line code (single basic block) and
simple control flow (loops via back-edges, if/else via branches).
"""

import re
import struct
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# LLVM intrinsic → WGSL function mapping
# ---------------------------------------------------------------------------

LLVM_UNARY_INTRINSICS = {
    'llvm.fabs': 'abs',
    'llvm.exp': 'exp',
    'llvm.exp2': 'exp2',
    'llvm.log': 'log',
    'llvm.log2': 'log2',
    'llvm.sqrt': 'sqrt',
    'llvm.sin': 'sin',
    'llvm.cos': 'cos',
    'llvm.ceil': 'ceil',
    'llvm.floor': 'floor',
    'llvm.round': 'round',
    'llvm.roundeven': 'round',
    'llvm.trunc': 'trunc',
    'llvm.rint': 'round',
    'llvm.nearbyint': 'round',
    'llvm.fabs': 'abs',
}

LLVM_BINARY_INTRINSICS = {
    'llvm.maxnum': 'max',
    'llvm.minnum': 'min',
    'llvm.smin': 'min',
    'llvm.smax': 'max',
    'llvm.umin': 'min',
    'llvm.umax': 'max',
    'llvm.pow': 'pow',
    'llvm.copysign': 'sign',
}

LLVM_TERNARY_INTRINSICS = {
    'llvm.fma': 'fma',
    'llvm.fmuladd': 'fma',
}


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

TRITON_TYPE_TO_WGSL = {
    'fp16': 'f16',
    'fp32': 'f32',
    'f32': 'f32',
    'fp64': 'f64',
    'i8': 'i32',  # WGSL has no i8; promote
    'i16': 'i32',  # WGSL has no i16; promote
    'i32': 'i32',
    'i64': 'i32',  # WGSL has no i64; truncate (sufficient for indices)
    'u8': 'u32',
    'u16': 'u32',
    'u32': 'u32',
    'u64': 'u32',
}

LLVM_TYPE_TO_WGSL = {
    'float': 'f32',
    'double': 'f64',
    'half': 'f16',
    'i32': 'i32',
    'i64': 'i32',
    'i16': 'i32',
    'i8': 'i32',
    'i1': 'bool',
}

LLVM_TYPE_TO_BYTES = {
    'float': 4,
    'double': 8,
    'half': 2,
    'i32': 4,
    'i64': 8,
    'i16': 2,
    'i8': 1,
    'i1': 1,
}


def triton_ptr_elem_type(sig_type: str) -> str:
    """Extract WGSL element type from Triton pointer type like '*fp32'."""
    assert sig_type.startswith('*'), f"Not a pointer type: {sig_type}"
    elem = sig_type[1:]
    return TRITON_TYPE_TO_WGSL.get(elem, 'f32')


def triton_scalar_wgsl_type(sig_type: str) -> str:
    """Convert Triton scalar type like 'i32' to WGSL type."""
    return TRITON_TYPE_TO_WGSL.get(sig_type, 'i32')


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BufferBinding:
    """A storage buffer binding in the WGSL shader."""
    binding: int
    name: str
    elem_type: str  # WGSL element type: f32, i32, etc.
    access: str  # 'read' or 'read_write'


@dataclass
class ParamField:
    """A scalar parameter field in the uniform/params struct."""
    name: str
    wgsl_type: str
    byte_size: int


@dataclass
class GepInfo:
    """Tracking info for a GEP (pointer) value."""
    buffer_arg_idx: int  # Which function arg (buffer binding index)
    offset_expr: str  # WGSL expression for the element offset
    is_uniform: bool = True  # Whether offset is the same across all threads


@dataclass
class IRBasicBlock:
    """A parsed LLVM IR basic block."""
    label: str
    phis: List[str]         # phi instructions
    body: List[str]         # non-phi, non-terminator instructions
    terminator: str         # branch or ret instruction


@dataclass
class TranslationResult:
    """Result of LLVM IR → WGSL translation."""
    wgsl: str
    workgroup_size: int
    buffer_bindings: List[BufferBinding]
    param_fields: List[ParamField]
    kernel_name: str


# ---------------------------------------------------------------------------
# LLVM IR Parser & WGSL Generator
# ---------------------------------------------------------------------------

class LLVMToWGSL:
    """
    Translates LLVM IR text from Triton's WebGPU backend into WGSL.

    Handles single-BB (straight-line) kernels and simple multi-BB kernels
    with structured control flow.
    """

    def __init__(self, llir: str, signature: dict,
                 num_warps: int = 4, warp_size: int = 32,
                 use_native_subgroups: bool = False):
        """
        Args:
            llir: LLVM IR string from Triton compilation
            signature: Triton parameter signature (excluding constexprs).
                       e.g. {'x_ptr': '*fp32', 'n_elements': 'i32'}
            num_warps: Warps per workgroup
            warp_size: Threads per warp
            use_native_subgroups: Use native WGSL subgroupShuffleXor instead
                                  of shared-memory emulation (requires the
                                  WebGPU Subgroups feature on the adapter)
        """
        self.llir = llir
        self.signature = signature
        self.num_warps = num_warps
        self.warp_size = warp_size
        self.workgroup_size = num_warps * warp_size
        self.use_native_subgroups = use_native_subgroups

        # State
        self.kernel_name = ""
        self.func_args: List[Tuple[str, str]] = []  # (llvm_type, arg_name)
        self.values: Dict[str, Tuple[str, str]] = {}  # ssa_name → (wgsl_expr, wgsl_type)
        self.gep_info: Dict[str, GepInfo] = {}  # ssa_name → GEP provenance
        self.stored_buffers: set = set()  # Set of buffer arg indices that are written to
        self.atomic_buffers: set = set()  # Set of buffer arg indices with atomic operations
        self.needs_subgroups: bool = False  # Whether WGSL needs 'enable subgroups;'
        self.needs_f16: bool = False  # Whether WGSL needs 'enable f16;'
        self.needs_shuffle_scratch: bool = False  # Whether we need shared mem for shuffle emulation
        self.needs_smem: bool = False  # Whether we need workgroup shared memory (addrspace(3))
        self.smem_bytes: int = 0  # Total shared memory size in bytes (auto-detected)
        self.smem_gep_info: Dict[str, str] = {}  # SSA → byte offset expression for shared mem GEPs
        self.masked_values: Dict[str, str] = {}  # SSA → condition for select w/ undef
        self.struct_fields: Dict[str, Dict[int, str]] = {}  # SSA → {field_idx: value_expr} for struct agg
        self.vector_elements: Dict[str, Dict[int, Tuple[str, str]]] = {}  # SSA → {idx: (expr, type)} for vectors
        self.uniform_values: set = set()  # SSA names that are thread-uniform (same across all threads in WG)

        # Binding info
        self.buffer_bindings: List[BufferBinding] = []
        self.param_fields: List[ParamField] = []
        self.sig_names: List[str] = []  # Ordered param names from signature
        self.sig_types: List[str] = []  # Ordered param types from signature

        # Separate pointer vs scalar args
        self.ptr_arg_indices: List[int] = []  # func_arg indices that are pointers
        self.scalar_arg_indices: List[int] = []  # func_arg indices that are scalars
        # Map: func_arg_index → binding_index (for pointers)
        self.arg_to_binding: Dict[int, int] = {}
        # Map: func_arg_index → param field name (for scalars)
        self.arg_to_param: Dict[int, str] = {}

    def translate(self) -> TranslationResult:
        """Run the full translation pipeline."""
        self._parse_signature()
        self._parse_function_header()
        self._classify_args()
        self._prescan_stores()
        self._prescan_smem_size()
        self._build_bindings()
        self._detect_f16()

        body_lines = self._extract_body()
        wgsl_stmts = self._translate_instructions(body_lines)
        wgsl_code = self._emit_wgsl(wgsl_stmts)

        return TranslationResult(
            wgsl=wgsl_code,
            workgroup_size=self.workgroup_size,
            buffer_bindings=self.buffer_bindings,
            param_fields=self.param_fields,
            kernel_name=self.kernel_name,
        )

    # -------------------------------------------------------------------
    # Step 1: Parse Triton signature
    # -------------------------------------------------------------------

    def _parse_signature(self):
        """Extract ordered parameter names and types from Triton signature.

        If no signature was provided, auto-infer from the LLVM IR function
        header: ``ptr addrspace(1)`` → ``*fp32`` (default), scalars keep
        their LLVM type.
        """
        if self.signature:
            for name, ty in self.signature.items():
                if ty == 'constexpr':
                    continue
                self.sig_names.append(name)
                self.sig_types.append(ty)
        # If signature is empty, we defer to _classify_args which
        # handles the no-signature case.

    # -------------------------------------------------------------------
    # Step 2: Parse LLVM function header
    # -------------------------------------------------------------------

    def _parse_function_header(self):
        """Extract kernel name and function arguments from LLVM IR."""
        # Match: define void @kernel_name(type %0, type %1, ...)
        # Use .*? instead of [^)]* to handle nested parens in addrspace(1)
        pattern = r'define\s+void\s+@(\w+)\((.+?)\)\s*\{'
        m = re.search(pattern, self.llir, re.DOTALL)
        if not m:
            raise ValueError("Could not find function definition in LLVM IR")

        self.kernel_name = m.group(1)
        args_str = m.group(2)

        # Parse each argument
        # Examples: "ptr addrspace(1) %0", "i32 %3"
        arg_pattern = r'((?:ptr\s+addrspace\(\d+\))|(?:i\d+)|(?:float|double|half))\s+(%\d+)'
        for am in re.finditer(arg_pattern, args_str):
            llvm_type = am.group(1).strip()
            arg_name = am.group(2)
            self.func_args.append((llvm_type, arg_name))

    # -------------------------------------------------------------------
    # Step 3: Classify args as pointers or scalars
    # -------------------------------------------------------------------

    def _classify_args(self):
        """Map LLVM function args to pointer (buffer) or scalar (param) args."""
        sig_idx = 0
        for i, (llvm_type, _arg_name) in enumerate(self.func_args):
            is_ptr = 'addrspace' in llvm_type
            if sig_idx < len(self.sig_types):
                sig_type = self.sig_types[sig_idx]
                if is_ptr and sig_type.startswith('*'):
                    self.ptr_arg_indices.append(i)
                    sig_idx += 1
                elif not is_ptr and not sig_type.startswith('*'):
                    self.scalar_arg_indices.append(i)
                    sig_idx += 1
                else:
                    # Extra internal arg (doesn't match signature)
                    if is_ptr:
                        self.ptr_arg_indices.append(i)
                    else:
                        self.scalar_arg_indices.append(i)
            else:
                # Extra args beyond the signature (internal Triton args)
                if is_ptr:
                    self.ptr_arg_indices.append(i)
                elif not is_ptr:
                    self.scalar_arg_indices.append(i)

        # Auto-populate sig_names/sig_types when signature was empty
        if not self.sig_names:
            for i, (llvm_type, arg_name) in enumerate(self.func_args):
                is_ptr = 'addrspace' in llvm_type
                name = f'arg{i}'
                if is_ptr:
                    self.sig_names.append(name)
                    self.sig_types.append('*fp32')  # default
                else:
                    wgsl_t = LLVM_TYPE_TO_WGSL.get(llvm_type, 'i32')
                    triton_t = 'i32' if wgsl_t in ('i32', 'u32') else 'fp32'
                    self.sig_names.append(name)
                    self.sig_types.append(triton_t)

        # Mark scalar function arguments as uniform (same for all threads)
        for i in self.scalar_arg_indices:
            self.uniform_values.add(f'%{i}')

    # -------------------------------------------------------------------
    # Step 4: Pre-scan for stores to determine buffer access modes
    # -------------------------------------------------------------------

    def _prescan_stores(self):
        """Scan the LLVM IR body to find which buffers are stored to."""
        # Find all store instructions and trace which buffer arg they write to
        # Pattern: store T val, ptr addrspace(1) %ptr_ssa
        # val can be %ssa, literal, hex, or undef
        store_pattern = r'store\s+\S+\s+(?:%\d+|[\d.e+-]+|0x[\da-fA-F]+|undef|true|false).*ptr\s+addrspace\(1\)\s+(%\d+)'
        gep_pattern = r'(%\d+)\s*=\s*getelementptr\s+(?:inbounds\s+)?\S+,\s*ptr\s+addrspace\(1\)\s+(%\d+|\d+)'

        # First, build a map of GEP destination → base argument
        gep_to_base = {}
        for m in re.finditer(gep_pattern, self.llir):
            gep_dst = m.group(1)
            base = m.group(2)
            # If base is a function arg like %0, %1, etc.
            if base.startswith('%'):
                try:
                    arg_idx = int(base[1:])
                    if arg_idx < len(self.func_args):
                        gep_to_base[gep_dst] = arg_idx
                except ValueError:
                    pass

        # Propagate chained GEPs: if base is a known GEP, inherit its arg
        changed = True
        while changed:
            changed = False
            for m in re.finditer(gep_pattern, self.llir):
                gep_dst = m.group(1)
                base = m.group(2)
                if gep_dst not in gep_to_base and base in gep_to_base:
                    gep_to_base[gep_dst] = gep_to_base[base]
                    changed = True

        # Then check which GEP destinations are used in stores
        for m in re.finditer(store_pattern, self.llir):
            ptr_ssa = m.group(1)
            if ptr_ssa in gep_to_base:
                self.stored_buffers.add(gep_to_base[ptr_ssa])
            else:
                # Check if ptr_ssa is a direct function arg (scalar store to buffer)
                if ptr_ssa.startswith('%'):
                    try:
                        arg_idx = int(ptr_ssa[1:])
                        if arg_idx < len(self.func_args):
                            llvm_type = self.func_args[arg_idx][0]
                            if 'addrspace' in llvm_type:
                                self.stored_buffers.add(arg_idx)
                    except ValueError:
                        pass

        # Also scan for atomicrmw instructions and mark those buffers
        # Pattern: atomicrmw OP ptr addrspace(1) %ptr_ssa, TYPE %val ...
        atomic_pattern = r'atomicrmw\s+\w+\s+ptr\s+addrspace\(1\)\s+(%\d+)'
        for m in re.finditer(atomic_pattern, self.llir):
            ptr_ssa = m.group(1)
            if ptr_ssa in gep_to_base:
                arg_idx = gep_to_base[ptr_ssa]
                self.atomic_buffers.add(arg_idx)
                self.stored_buffers.add(arg_idx)  # atomics are read_write

    def _prescan_smem_size(self):
        """Scan LLVM IR to determine the required shared memory size in bytes.

        Detects patterns like:
          - getelementptr i8, ptr addrspace(3) @global_smem, i32 <offset>
          - getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 <base>), i32 <offset>
          - store ... ptr addrspace(3) <ptr>, with byte offset tracking
        """
        max_static_offset = 0
        max_dynamic_slots = 0

        # Find all constant offsets into @global_smem (including nested GEP bases)
        # Pattern 1: direct GEP with constant offset
        for m in re.finditer(
            r'getelementptr\s+(?:inbounds\s+)?i8,\s*ptr\s+addrspace\(3\)\s+@global_smem,\s*i(?:32|64)\s+(\d+)',
            self.llir
        ):
            off = int(m.group(1))
            if off > max_static_offset:
                max_static_offset = off

        # Pattern 2: nested GEP base offset (i64 constant) in constant expression
        for m in re.finditer(
            r'getelementptr\s*\(i8,\s*ptr\s+addrspace\(3\)\s+@global_smem,\s*i64\s+(\d+)\)',
            self.llir
        ):
            base_off = int(m.group(1))
            if base_off > max_static_offset:
                max_static_offset = base_off

        # The max offset tells us how far into smem code accesses.
        # Add a margin of 1024 bytes for the data stored at the highest offset.
        if max_static_offset > 0:
            self.smem_bytes = max_static_offset + 1024
        elif self.needs_smem:
            # Fallback: num_warps * 4 bytes (for simple reductions)
            self.smem_bytes = self.num_warps * 4
        else:
            self.smem_bytes = 0

    # -------------------------------------------------------------------
    # Step 5: Build buffer bindings and param struct
    # -------------------------------------------------------------------

    def _build_bindings(self):
        """Create WGSL binding declarations."""
        binding_idx = 0

        # Detect which LLVM args are actually used in the IR body.
        # Internal (extra) pointer args that are never referenced can be
        # safely omitted; binding them with dummy buffers can cause D3D12
        # issues on some drivers.
        used_args: set = set()
        # Extract just the function body (everything after the opening '{')
        body_start = self.llir.find('{')
        ir_body = self.llir[body_start:] if body_start >= 0 else self.llir
        for arg_idx, (_llvm_type, _arg_name) in enumerate(self.func_args):
            pattern = rf'(?<!\d)%{arg_idx}(?!\d)'
            if re.search(pattern, ir_body):
                used_args.add(arg_idx)

        # Create buffer bindings for pointer args
        for i, arg_idx in enumerate(self.ptr_arg_indices):
            # Determine element type and name
            if i < len(self.sig_names) and self.sig_types[i].startswith('*'):
                name = self.sig_names[i]
                elem_type = triton_ptr_elem_type(self.sig_types[i])
            else:
                name = f"_internal_{arg_idx}"
                elem_type = 'f32'
                # Skip unused internal args
                if arg_idx not in used_args:
                    continue

            access = 'read_write' if arg_idx in self.stored_buffers else 'read'

            self.buffer_bindings.append(BufferBinding(
                binding=binding_idx,
                name=name,
                elem_type=elem_type,
                access=access,
            ))
            self.arg_to_binding[arg_idx] = binding_idx
            binding_idx += 1

        # Create param fields for scalar args
        param_binding_idx = binding_idx
        for i, arg_idx in enumerate(self.scalar_arg_indices):
            # Find the matching signature entry
            scalar_count = 0
            for si, st in enumerate(self.sig_types):
                if not st.startswith('*'):
                    if scalar_count == i:
                        name = self.sig_names[si]
                        wgsl_type = triton_scalar_wgsl_type(st)
                        self.param_fields.append(ParamField(
                            name=name, wgsl_type=wgsl_type,
                            byte_size=4,  # i32 or f32
                        ))
                        self.arg_to_param[arg_idx] = name
                        break
                    scalar_count += 1

    # -------------------------------------------------------------------
    # Step 5b: Detect f16 usage
    # -------------------------------------------------------------------

    def _detect_f16(self):
        """Set needs_f16 if any buffer uses f16 or LLVM IR contains half ops."""
        for bb in self.buffer_bindings:
            if bb.elem_type == 'f16':
                self.needs_f16 = True
                return
        # Also scan LLVM IR for half type usage
        if re.search(r'\bhalf\b', self.llir):
            self.needs_f16 = True

    # -------------------------------------------------------------------
    # Step 6: Extract function body
    # -------------------------------------------------------------------

    def _extract_body(self) -> List[str]:
        """Extract instruction lines from the function body."""
        # Find the function body between { and }
        # Use .*? for args to handle nested parens like addrspace(1)
        func_match = re.search(
            r'define\s+void\s+@\w+\(.*?\)\s*\{(.*?)\n\}',
            self.llir, re.DOTALL
        )
        if not func_match:
            raise ValueError("Could not extract function body")

        body = func_match.group(1)
        lines = []
        for line in body.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            # Remove inline comments
            if ';' in line:
                line = line[:line.index(';')].strip()
            if line:
                lines.append(line)
        return lines

    # -------------------------------------------------------------------
    # Step 7: Parse basic blocks and translate instructions
    # -------------------------------------------------------------------

    def _parse_basic_blocks(self, lines: List[str]) -> List[IRBasicBlock]:
        """Parse LLVM IR lines into basic blocks."""
        blocks = []
        current_label = 'entry'
        current_phis = []
        current_body = []
        current_terminator = None

        for line in lines:
            # Label definition: "9:" or "14:  ..." (already stripped of comments)
            label_match = re.match(r'^(\d+):$', line)
            if label_match:
                # Save previous block
                if current_body or current_phis or current_terminator:
                    blocks.append(IRBasicBlock(
                        label=current_label,
                        phis=current_phis,
                        body=current_body,
                        terminator=current_terminator or '',
                    ))
                current_label = label_match.group(1)
                current_phis = []
                current_body = []
                current_terminator = None
                continue

            # Terminator instructions
            if line == 'ret void' or line.startswith('br '):
                current_terminator = line
                continue

            # Phi instruction
            if '= phi ' in line:
                current_phis.append(line)
                continue

            current_body.append(line)

        # Save last block
        if current_body or current_phis or current_terminator:
            blocks.append(IRBasicBlock(
                label=current_label,
                phis=current_phis,
                body=current_body,
                terminator=current_terminator or 'ret void',
            ))

        return blocks

    def _translate_instructions(self, lines: List[str]) -> List[str]:
        """Translate LLVM IR instructions to WGSL statements."""
        blocks = self._parse_basic_blocks(lines)

        if len(blocks) <= 1:
            # Single basic block — existing fast path
            stmts = []
            for line in lines:
                result = self._translate_one(line)
                if result:
                    stmts.append(result)
            return stmts

        # Multi-basic-block: reconstruct structured control flow
        return self._translate_multi_block(blocks)

    def _translate_multi_block(self, blocks: List['IRBasicBlock']) -> List[str]:
        """Translate multi-basic-block LLVM IR into structured WGSL."""
        stmts = []
        block_map = {b.label: b for b in blocks}
        visited = set()

        def find_successors(block):
            """Return the list of successor labels from the terminator."""
            t = block.terminator
            if t == 'ret void' or not t:
                return []
            m = re.match(r'br i1 %\d+, label %(\w+), label %(\w+)', t)
            if m:
                return [m.group(1), m.group(2)]
            m = re.match(r'br label %(\w+)', t)
            if m:
                return [m.group(1)]
            return []

        def find_merge_point(true_label, false_label):
            """Find the merge point of a diamond pattern via BFS."""
            # Follow successors from each branch until paths converge
            reachable_from_true = set()
            queue = [true_label]
            while queue:
                lbl = queue.pop(0)
                if lbl in reachable_from_true:
                    continue
                reachable_from_true.add(lbl)
                b = block_map.get(lbl)
                if b:
                    for s in find_successors(b):
                        queue.append(s)

            # Now search from false_label for first block also in true's reachable set
            queue = [false_label]
            seen = set()
            while queue:
                lbl = queue.pop(0)
                if lbl in seen:
                    continue
                seen.add(lbl)
                if lbl in reachable_from_true and lbl != true_label and lbl != false_label:
                    return lbl
                # Also check if true_label IS the merge point (else-then pattern)
                b = block_map.get(lbl)
                if b:
                    for s in find_successors(b):
                        if s in reachable_from_true:
                            return s
                        queue.append(s)

            # If false block jumps directly to true_label, true_label is the merge
            if true_label in reachable_from_true:
                return true_label
            return None

        def resolve_phi_value(phi_line, from_label):
            """Extract the value from a phi instruction for a given predecessor label."""
            # phi type [val1, %label1], [val2, %label2], ...
            pairs = re.findall(r'\[\s*(%\d+|\d+|[\d.e+-]+|0x[\da-fA-F]+|undef|zeroinitializer),\s*%(\w+)\s*\]', phi_line)
            for val, lbl in pairs:
                if lbl == from_label:
                    return val
            return None

        def phi_dst_and_type(phi_line):
            """Extract destination SSA and type from a phi instruction."""
            m = re.match(r'(%\d+)\s*=\s*phi\s+(\S+)', phi_line)
            if m:
                return m.group(1), m.group(2)
            return None, None

        def process_block_body(block, into_stmts):
            """Translate body instructions of a block, appending to stmts."""
            for line in block.body:
                result = self._translate_one(line)
                if result:
                    into_stmts.append(result)

        def collect_predecessors(label):
            """Return labels of blocks that branch to the given label."""
            preds = []
            for b in blocks:
                succs = find_successors(b)
                if label in succs:
                    preds.append(b.label)
            return preds

        def is_loop_header(target_block, entry_label):
            """Check if target_block is a loop header (has a back-edge from a body block)."""
            t = target_block.terminator
            if not t:
                return False, None, None, None
            cond = re.match(r'br i1 (%\d+), label %(\w+), label %(\w+)', t)
            if not cond:
                return False, None, None, None

            cond_ssa = cond.group(1)
            true_lbl = cond.group(2)
            false_lbl = cond.group(3)

            # Check if the true branch has a back-edge to this header
            true_blk = block_map.get(true_lbl)
            if true_blk:
                true_term = true_blk.terminator or ''
                back = re.match(r'br label %(\w+)', true_term)
                if back and back.group(1) == target_block.label:
                    # True branch is the loop body, false is exit
                    return True, cond_ssa, true_lbl, false_lbl

            # Check if there's a chain: true_blk contains if/else that eventually
            # branches back to header
            if true_blk:
                # Check all reachable blocks from true_lbl (within 3 hops)
                check_queue = [true_lbl]
                check_seen = set()
                for _ in range(5):
                    if not check_queue:
                        break
                    lbl = check_queue.pop(0)
                    if lbl in check_seen or lbl == false_lbl:
                        continue
                    check_seen.add(lbl)
                    blk = block_map.get(lbl)
                    if blk:
                        for s in find_successors(blk):
                            if s == target_block.label:
                                return True, cond_ssa, true_lbl, false_lbl
                            if s not in check_seen:
                                check_queue.append(s)

            # Check false branch as body (inverted condition)
            false_blk = block_map.get(false_lbl)
            if false_blk:
                false_term = false_blk.terminator or ''
                back = re.match(r'br label %(\w+)', false_term)
                if back and back.group(1) == target_block.label:
                    # False branch is body, true is exit — condition inverted
                    return True, cond_ssa, false_lbl, true_lbl

            return False, None, None, None

        def parse_struct_phi_type(phi_line):
            """Extract struct field types from a phi of struct type.
            Returns (is_struct, field_types_list)."""
            # e.g. %62 = phi { float, float } [...]
            m = re.match(r'%\d+\s*=\s*phi\s+\{([^}]+)\}', phi_line)
            if m:
                field_types = [t.strip() for t in m.group(1).split(',')]
                return True, field_types
            return False, []

        def emit_loop(header_block, entry_block, cond_ssa,
                      body_label, exit_label):
            """Emit a WGSL loop construct for a natural loop."""
            entry_label = entry_block.label
            if entry_label == 'entry':
                entry_label = str(len(self.func_args))

            # Declare and initialize phi variables from entry values
            for phi_line in header_block.phis:
                dst, ty = phi_dst_and_type(phi_line)
                if not dst:
                    continue

                is_struct, field_types = parse_struct_phi_type(phi_line)
                if is_struct:
                    # Struct phi: declare var for each field
                    init_val = resolve_phi_value(phi_line, entry_label)
                    for fi, ft in enumerate(field_types):
                        wt = LLVM_TYPE_TO_WGSL.get(ft, 'f32')
                        var_name = f'{self._var(dst)}_{fi}'
                        if init_val == 'zeroinitializer' or init_val == 'undef' or init_val is None:
                            stmts.append(f'var {var_name}: {wt} = {self._zero_val(wt)};')
                        elif init_val.startswith('%') and init_val in self.struct_fields:
                            stmts.append(f'var {var_name}: {wt} = {self.struct_fields[init_val].get(fi, self._zero_val(wt))};')
                        else:
                            stmts.append(f'var {var_name}: {wt} = {self._zero_val(wt)};')
                        self.values[f'{dst}_{fi}'] = (var_name, wt)
                    # Register the struct fields dict
                    self.struct_fields[dst] = {fi: f'{self._var(dst)}_{fi}'
                                               for fi in range(len(field_types))}
                else:
                    if 'ptr' in (ty or ''):
                        continue  # Skip pointer phis
                    wgsl_ty = LLVM_TYPE_TO_WGSL.get(ty, 'i32')
                    init_val = resolve_phi_value(phi_line, entry_label)
                    init_expr = self._operand(init_val, wgsl_ty) if init_val else self._zero_val(wgsl_ty)
                    stmts.append(f'var {self._var(dst)}: {wgsl_ty} = {init_expr};')
                    self.values[dst] = (self._var(dst), wgsl_ty)

            # Process header body (compute condition, etc.)
            # First translate header's body instructions to compute the condition
            header_body_stmts = []
            process_block_body(header_block, header_body_stmts)

            # Emit WGSL loop
            stmts.append('loop {')

            # Header body inside loop (condition computation)
            for s in header_body_stmts:
                stmts.append(f'    {s}')

            # Break condition
            cond_expr = self._operand(cond_ssa, 'bool')
            # Determine which direction: if cond is true → body, false → exit
            # Then we break when !cond
            stmts.append(f'    if !{cond_expr} {{ break; }}')

            # Process loop body blocks
            body_block = block_map.get(body_label)
            if body_block:
                visited.add(body_label)
                # Handle inner if/else within the body
                body_stmts = []
                self._translate_loop_body(body_block, header_block.label,
                                          block_map, visited, body_stmts)
                for s in body_stmts:
                    stmts.append(f'    {s}')

            # Update phi variables from back-edge (continuing block)
            # Find which block provides the back-edge values
            back_label = self._find_back_edge_label(
                body_label, header_block.label, block_map, visited)

            for phi_line in header_block.phis:
                dst, ty = phi_dst_and_type(phi_line)
                if not dst:
                    continue
                is_struct, field_types = parse_struct_phi_type(phi_line)
                if is_struct:
                    back_val = resolve_phi_value(phi_line, back_label)
                    if back_val and back_val in self.struct_fields:
                        for fi, ft in enumerate(field_types):
                            wt = LLVM_TYPE_TO_WGSL.get(ft, 'f32')
                            var_name = f'{self._var(dst)}_{fi}'
                            src_expr = self.struct_fields[back_val].get(fi, self._zero_val(wt))
                            stmts.append(f'    {var_name} = {src_expr};')
                else:
                    if 'ptr' in (ty or ''):
                        continue
                    wgsl_ty = LLVM_TYPE_TO_WGSL.get(ty, 'i32')
                    back_val = resolve_phi_value(phi_line, back_label)
                    if back_val:
                        update_expr = self._operand(back_val, wgsl_ty)
                        stmts.append(f'    {self._var(dst)} = {update_expr};')

            stmts.append('}')

            # Process exit block
            visited.add(header_block.label)
            exit_block = block_map.get(exit_label)
            if exit_block and exit_label not in visited:
                visited.add(exit_label)
                process_block_body(exit_block, stmts)
                # Follow exit terminator
                process_block_terminator(exit_block)

        def process_block_terminator(block):
            """Process just the terminator of a block (follow unconditional branches).

            Includes loop detection and phi resolution so that sequential loops
            (e.g., mean loop followed by variance loop) are properly structured.
            """
            t = block.terminator
            if not t or t == 'ret void':
                return
            uncond = re.match(r'br label %(\w+)', t)
            if uncond:
                tgt_label = uncond.group(1)
                tgt = block_map.get(tgt_label)
                if tgt and tgt_label not in visited:
                    # Check if target is a loop header
                    is_loop, loop_cond, loop_body, loop_exit = is_loop_header(tgt, block.label)
                    if is_loop:
                        emit_loop(tgt, block, loop_cond, loop_body, loop_exit)
                        return

                    # Not a loop — process phis at target and continue
                    for phi in tgt.phis:
                        dst, ty = phi_dst_and_type(phi)
                        if dst and 'ptr' not in ty:
                            is_struct, field_types = parse_struct_phi_type(phi)
                            if is_struct:
                                init_val = resolve_phi_value(phi, block.label)
                                for fi, ft in enumerate(field_types):
                                    wt = LLVM_TYPE_TO_WGSL.get(ft, 'f32')
                                    var_name = f'{self._var(dst)}_{fi}'
                                    if init_val and init_val in self.struct_fields:
                                        src_expr = self.struct_fields[init_val].get(fi, self._zero_val(wt))
                                        stmts.append(f'let {var_name}: {wt} = {src_expr};')
                                    elif init_val == 'zeroinitializer' or init_val == 'undef' or not init_val:
                                        stmts.append(f'let {var_name}: {wt} = {self._zero_val(wt)};')
                                    else:
                                        stmts.append(f'let {var_name}: {wt} = {self._zero_val(wt)};')
                                    self.values[f'{dst}_{fi}'] = (var_name, wt)
                                self.struct_fields[dst] = {fi: f'{self._var(dst)}_{fi}'
                                                           for fi in range(len(field_types))}
                            else:
                                val = resolve_phi_value(phi, block.label)
                                if val:
                                    wgsl_ty = LLVM_TYPE_TO_WGSL.get(ty, 'i32')
                                    self.values[dst] = (self._var(dst), wgsl_ty)
                                    val_expr = self._operand(val, wgsl_ty)
                                    stmts.append(f'let {self._var(dst)}: {wgsl_ty} = {val_expr};')
                    process_block(tgt)

        def process_block(block):
            """Process a block and follow its control flow."""
            if block.label in visited:
                return
            visited.add(block.label)

            # Process body
            process_block_body(block, stmts)

            # Handle terminator
            t = block.terminator
            if t == 'ret void' or not t:
                return

            # Unconditional branch
            uncond = re.match(r'br label %(\w+)', t)
            if uncond:
                tgt_label = uncond.group(1)
                tgt = block_map.get(tgt_label)
                if tgt and tgt_label not in visited:
                    # Check if target is a loop header
                    is_loop, loop_cond, loop_body, loop_exit = is_loop_header(tgt, block.label)
                    if is_loop:
                        emit_loop(tgt, block, loop_cond, loop_body, loop_exit)
                        return

                    # Not a loop — process phis at target and continue
                    for phi in tgt.phis:
                        dst, ty = phi_dst_and_type(phi)
                        if dst and 'ptr' not in ty:
                            val = resolve_phi_value(phi, block.label)
                            if val:
                                wgsl_ty = LLVM_TYPE_TO_WGSL.get(ty, 'i32')
                                self.values[dst] = (self._var(dst), wgsl_ty)
                                val_expr = self._operand(val, wgsl_ty)
                                stmts.append(f'let {self._var(dst)}: {wgsl_ty} = {val_expr};')
                    process_block(tgt)
                return

            # Conditional branch
            cond_match = re.match(r'br i1 (%\d+), label %(\w+), label %(\w+)', t)
            if cond_match:
                cond_ssa = cond_match.group(1)
                true_label = cond_match.group(2)
                false_label = cond_match.group(3)
                self._translate_conditional_branch(
                    cond_ssa, true_label, false_label,
                    block, block_map, stmts, visited
                )

        process_block(blocks[0])
        return stmts

    def _translate_conditional_branch(
        self, cond_ssa, true_label, false_label,
        entry_block, block_map, stmts, visited
    ):
        """Translate a conditional branch with phi resolution into WGSL if/else."""
        cond_expr = self._operand(cond_ssa, 'bool')

        # Find merge point
        # Pattern: one branch may jump directly to the other (forming a triangle/diamond)
        true_block = block_map.get(true_label)
        false_block = block_map.get(false_label)

        if not true_block or not false_block:
            stmts.append(f'// Unresolved branch: {cond_ssa}')
            return

        # Determine which block is the merge point
        true_succs = set()
        m = re.match(r'br label %(\w+)', true_block.terminator or '')
        if m:
            true_succs.add(m.group(1))
        false_succs = set()
        m = re.match(r'br label %(\w+)', false_block.terminator or '')
        if m:
            false_succs.add(m.group(1))

        merge_label = None
        # Pattern 1: Both branches converge to same block
        common = true_succs & false_succs
        if common:
            merge_label = common.pop()
        # Pattern 2: false block jumps to true_label (true_label is merge)
        elif true_label in false_succs:
            merge_label = true_label
        # Pattern 3: true block jumps to false_label (false_label is merge)
        elif false_label in true_succs:
            merge_label = false_label

        if not merge_label:
            # Fallback: just process both blocks sequentially
            stmts.append(f'// Complex control flow - linearized')
            visited.add(true_label)
            visited.add(false_label)
            for line in true_block.body:
                r = self._translate_one(line)
                if r:
                    stmts.append(r)
            for line in false_block.body:
                r = self._translate_one(line)
                if r:
                    stmts.append(r)
            return

        merge_block = block_map.get(merge_label)

        # Determine which predecessor labels feed into the merge phi nodes
        # entry_label is the label of the block that branches
        entry_label = entry_block.label
        if entry_label == 'entry':
            # Find the actual numeric label for entry (usually %6 from the preds comments)
            # The entry block number is the arg count for the function
            entry_label = str(len(self.func_args))

        # Collect phi nodes at the merge point
        phi_declarations = []
        phi_true_assigns = []
        phi_false_assigns = []

        # Pre-resolve phis in non-merge blocks so their SSA values are known
        # (e.g., block 14's phi resolves %15 = %2 which is a func arg ptr)
        for block in [true_block, false_block]:
            if block.label == merge_label:
                continue
            for phi_line in block.phis:
                pm = re.match(r'(%\d+)\s*=\s*phi\s+(.+)', phi_line)
                if not pm:
                    continue
                phi_dst = pm.group(1)
                rest = pm.group(2)
                if 'ptr' in rest:
                    pairs = re.findall(
                        r'\[\s*(%\d+|\d+|[\d.e+-]+|0x[\da-fA-F]+|undef|zeroinitializer),\s*%(\w+)\s*\]',
                        phi_line
                    )
                    for val, lbl in pairs:
                        if val.startswith('%'):
                            try:
                                arg_idx = int(val[1:])
                                if arg_idx in self.arg_to_binding:
                                    self.gep_info[phi_dst] = GepInfo(
                                        buffer_arg_idx=arg_idx,
                                        offset_expr='0',
                                    )
                            except ValueError:
                                pass

        if merge_block:
            for phi_line in merge_block.phis:
                dst, ty = None, None
                m = re.match(r'(%\d+)\s*=\s*phi\s+(.+)', phi_line)
                if not m:
                    continue
                dst = m.group(1)
                rest = m.group(2)

                # Extract type (may be "ptr addrspace(1)" or "i32")
                is_ptr = 'ptr' in rest
                if is_ptr:
                    ty_str = 'ptr'
                    wgsl_type = 'i32'  # pointers get resolved via loads
                else:
                    ty_match = re.match(r'(\S+)\s+\[', rest)
                    ty_str = ty_match.group(1) if ty_match else 'i32'
                    wgsl_type = LLVM_TYPE_TO_WGSL.get(ty_str, 'i32')

                # Parse [value, %label] pairs
                pairs = re.findall(
                    r'\[\s*(%\d+|\d+|[\d.e+-]+|0x[\da-fA-F]+|undef|zeroinitializer),\s*%(\w+)\s*\]',
                    phi_line
                )

                # Determine which value comes from true path vs false path
                true_val = None
                false_val = None
                for val, lbl in pairs:
                    if lbl == entry_label:
                        # Direct from entry = true branch (or false, depends on pattern)
                        if merge_label == true_label:
                            # Merge IS the true block, so entry→true means direct (true path)
                            true_val = val
                        else:
                            true_val = val
                    elif lbl == false_label:
                        false_val = val
                    elif lbl == true_label:
                        true_val = val
                    else:
                        # Find which path this predecessor is on
                        # If the predecessor can reach true_label, it's on true path
                        false_val = val  # default

                if is_ptr:
                    # Phi on pointer: need to resolve buffer loads in each branch
                    # We'll handle the subsequent load from this phi result
                    # by setting up GEP info for each branch's value
                    # and then doing the load inside the if/else

                    # First, find the load that uses this phi result
                    load_dst = None
                    load_elem_type = 'i32'
                    if merge_block:
                        for body_line in merge_block.body:
                            load_match = re.match(
                                r'(%\d+)\s*=\s*load\s+(\S+),\s*ptr\s+addrspace\(1\)\s+' + re.escape(dst),
                                body_line
                            )
                            if load_match:
                                load_dst = load_match.group(1)
                                load_elem_type = load_match.group(2).rstrip(',')
                                break

                    if load_dst:
                        load_wgsl_type = LLVM_TYPE_TO_WGSL.get(load_elem_type, 'i32')
                        phi_declarations.append(
                            f'var {self._var(load_dst)}: {load_wgsl_type};'
                        )
                        self.values[load_dst] = (self._var(load_dst), load_wgsl_type)

                        # Generate load for true branch
                        true_load = self._resolve_ptr_load(true_val, load_wgsl_type)
                        false_load = self._resolve_ptr_load(false_val, load_wgsl_type)

                        phi_true_assigns.append(
                            f'{self._var(load_dst)} = {true_load};'
                        )
                        phi_false_assigns.append(
                            f'{self._var(load_dst)} = {false_load};'
                        )
                else:
                    # Scalar phi: declare var, assign in each branch
                    phi_declarations.append(
                        f'var {self._var(dst)}: {wgsl_type};'
                    )
                    self.values[dst] = (self._var(dst), wgsl_type)

                    if true_val:
                        phi_true_assigns.append(
                            f'{self._var(dst)} = {self._operand(true_val, wgsl_type)};'
                        )
                    if false_val:
                        phi_false_assigns.append(
                            f'{self._var(dst)} = {self._operand(false_val, wgsl_type)};'
                        )

        # Emit var declarations for phi results
        for decl in phi_declarations:
            stmts.append(decl)

        # Build true/false branch statement lists
        true_stmts = []
        false_stmts = []

        # Determine actual true vs false body based on merge pattern
        if merge_label == true_label:
            # True block IS the merge; false block has its own body
            # Entry→true (merge): assigns from entry_label
            # Entry→false→true (merge): assigns from false_label
            true_stmts.extend(phi_true_assigns)
            false_stmts.extend(phi_false_assigns)
            # Process false block body
            for line in false_block.body:
                r = self._translate_one(line)
                if r:
                    false_stmts.append(r)
        elif merge_label == false_label:
            # False block IS the merge; true block has its own body
            true_stmts.extend(phi_true_assigns)
            # Process true block body
            for line in true_block.body:
                r = self._translate_one(line)
                if r:
                    true_stmts.append(r)
            false_stmts.extend(phi_false_assigns)
        else:
            # Classic diamond: both branches converge to a separate merge
            true_stmts.extend(phi_true_assigns)
            for line in true_block.body:
                r = self._translate_one(line)
                if r:
                    true_stmts.append(r)
            false_stmts.extend(phi_false_assigns)
            for line in false_block.body:
                r = self._translate_one(line)
                if r:
                    false_stmts.append(r)

        # Emit if/else
        stmts.append(f'if {cond_expr} {{')
        for s in true_stmts:
            stmts.append(f'    {s}')
        if false_stmts:
            stmts.append(f'}} else {{')
            for s in false_stmts:
                stmts.append(f'    {s}')
        stmts.append('}')

        # Mark branches as visited (but not merge if it coincides)
        if true_label != merge_label:
            visited.add(true_label)
        if false_label != merge_label:
            visited.add(false_label)

        # Process merge block body (skip phi nodes and load from phi ptr)
        if merge_block and merge_label not in visited:
            visited.add(merge_label)
            for body_line in merge_block.body:
                # Skip loads from phi pointer results (already handled above)
                if any(re.search(re.escape(dst_ssa) + r'\b', body_line)
                       for phi_line in (merge_block.phis or [])
                       for dst_ssa in [re.match(r'(%\d+)', phi_line).group(1)]
                       if re.match(r'(%\d+)', phi_line)
                       and 'ptr' in phi_line):
                    continue
                r = self._translate_one(body_line)
                if r:
                    stmts.append(r)

            # Follow merge block's terminator
            t = merge_block.terminator
            if t and t != 'ret void':
                uncond = re.match(r'br label %(\w+)', t)
                if uncond:
                    next_label = uncond.group(1)
                    next_block = block_map.get(next_label)
                    if next_block and next_label not in visited:
                        visited.add(next_label)
                        # Recursively process remaining blocks
                        for body_line in next_block.body:
                            r = self._translate_one(body_line)
                            if r:
                                stmts.append(r)
                        # Follow chain of unconditional branches
                        nt = next_block.terminator
                        while nt and nt != 'ret void':
                            um = re.match(r'br label %(\w+)', nt)
                            if um:
                                nl = um.group(1)
                                nb = block_map.get(nl)
                                if nb and nl not in visited:
                                    visited.add(nl)
                                    for body_line in nb.body:
                                        r = self._translate_one(body_line)
                                        if r:
                                            stmts.append(r)
                                    nt = nb.terminator
                                else:
                                    break
                            else:
                                break

    def _translate_loop_body(self, body_block, header_label, block_map, visited, body_stmts):
        """Translate the body of a loop, handling inner if/else and chains.

        Processes body_block and follows its control flow until we hit
        the back-edge (branch back to header_label). Appends to body_stmts.
        """
        # Translate body instructions
        for line in body_block.body:
            result = self._translate_one(line)
            if result:
                body_stmts.append(result)

        # Follow terminator (except back-edge)
        t = body_block.terminator
        if not t or t == 'ret void':
            return

        # Unconditional branch
        uncond = re.match(r'br label %(\w+)', t)
        if uncond:
            tgt = uncond.group(1)
            if tgt == header_label:
                return  # Back-edge — stop
            tgt_block = block_map.get(tgt)
            if tgt_block and tgt not in visited:
                visited.add(tgt)
                self._translate_loop_body(tgt_block, header_label,
                                          block_map, visited, body_stmts)
            return

        # Conditional branch inside loop body (inner if/else)
        cond = re.match(r'br i1 (%\d+), label %(\w+), label %(\w+)', t)
        if cond:
            cond_ssa = cond.group(1)
            true_label = cond.group(2)
            false_label = cond.group(3)
            self._translate_conditional_branch(
                cond_ssa, true_label, false_label,
                body_block, block_map, body_stmts, visited
            )

    def _find_back_edge_label(self, body_label, header_label, block_map, visited):
        """Find the label of the block providing the back-edge to the loop header.

        Walk from body_label following unconditional branches until we find
        one that branches to header_label. Return its label.
        """
        seen = set()
        queue = [body_label]
        while queue:
            lbl = queue.pop(0)
            if lbl in seen:
                continue
            seen.add(lbl)
            blk = block_map.get(lbl)
            if not blk:
                continue
            t = blk.terminator
            if not t:
                continue
            # Check unconditional branch to header
            uncond = re.match(r'br label %(\w+)', t)
            if uncond and uncond.group(1) == header_label:
                return lbl
            # Follow branches
            cond = re.match(r'br i1 %\d+, label %(\w+), label %(\w+)', t)
            if cond:
                queue.append(cond.group(1))
                queue.append(cond.group(2))
            elif uncond:
                queue.append(uncond.group(1))
        return body_label  # Fallback

    def _resolve_ptr_load(self, ptr_val: str, wgsl_type: str) -> str:
        """Resolve a pointer value to a buffer load expression."""
        if ptr_val is None:
            return self._zero_val(wgsl_type)

        # Check if it's a function argument
        if ptr_val.startswith('%'):
            try:
                arg_idx = int(ptr_val[1:])
                if arg_idx in self.arg_to_binding:
                    binding_idx = self.arg_to_binding[arg_idx]
                    return f'buf{binding_idx}[0u]'
            except ValueError:
                pass

            # Check if it's a GEP result
            if ptr_val in self.gep_info:
                gep = self.gep_info[ptr_val]
                binding_idx = self.arg_to_binding.get(gep.buffer_arg_idx, 0)
                return f'buf{binding_idx}[u32({gep.offset_expr})]'

            # Check if it's a known value (like a phi-resolved pointer)
            if ptr_val in self.values:
                return self.values[ptr_val][0]

        return self._zero_val(wgsl_type)

    def _translate_one(self, line: str) -> Optional[str]:
        """Translate a single LLVM IR instruction to WGSL. Returns None for no-op."""

        # --- ret void ---
        if line.strip() == 'ret void':
            return None  # End of function, no explicit return needed

        # --- Assignment instructions: %dst = ... ---
        assign_match = re.match(r'(%\d+)\s*=\s*(.*)', line)
        if assign_match:
            dst = assign_match.group(1)
            rhs = assign_match.group(2).strip()
            return self._translate_assignment(dst, rhs)

        # --- store to shared memory (addrspace(3)) ---
        smem_store_match = re.match(
            r'store\s+(\S+)\s+(%\d+|[\d.e+-]+|0x[\da-fA-F]+|undef|true|false),\s*ptr\s+addrspace\(3\)\s+(%\d+)',
            line
        )
        if smem_store_match:
            ty = smem_store_match.group(1)
            val = smem_store_match.group(2)
            ptr = smem_store_match.group(3)
            return self._translate_smem_store(val, ptr, ty)

        # --- store instruction ---
        store_match = re.match(
            r'store\s+(\S+)\s+(%\d+|[\d.e+-]+|0x[\da-fA-F]+|undef|true|false),\s*ptr\s+addrspace\(1\)\s+(%\d+)',
            line
        )
        if store_match:
            ty = store_match.group(1)
            val = store_match.group(2)
            ptr = store_match.group(3)
            return self._translate_store(val, ptr, ty)

        # --- void calls (barriers, etc.) ---
        void_call_match = re.match(r'call\s+void\s+@(\w+)\(', line)
        if void_call_match:
            func_name = void_call_match.group(1)
            if func_name == '__spirv_ControlBarrier':
                return 'workgroupBarrier();'
            return None  # Ignore other void calls

        # Ignore other instructions (labels, metadata, etc.)
        return None

    def _translate_assignment(self, dst: str, rhs: str) -> Optional[str]:
        """Translate an assignment instruction."""

        # --- SPIR-V builtin calls ---
        builtin_match = re.match(
            r'call\s+i32\s+@__spirv_BuiltIn(\w+)\(i32\s+(\d+)\)', rhs
        )
        if builtin_match:
            builtin = builtin_match.group(1)
            axis = int(builtin_match.group(2))
            component = ['x', 'y', 'z'][axis]

            if builtin == 'WorkgroupId':
                expr = f'i32(_wg_id.{component})'
                self.uniform_values.add(dst)  # same across all threads
            elif builtin == 'LocalInvocationId':
                expr = f'i32(_lid.{component})'
                # NOT uniform — differs per thread
            elif builtin == 'NumWorkgroups':
                expr = f'i32(_num_wg.{component})'
                self.uniform_values.add(dst)  # same across all threads
            else:
                expr = f'0 /* unknown builtin: {builtin} */'

            self.values[dst] = (self._var(dst), 'i32')
            return f'let {self._var(dst)}: i32 = {expr};'

        # --- SubgroupShuffleXor call ---
        # Native path: subgroupShuffleXor() (requires Subgroups feature)
        # Fallback: emulate via workgroup shared memory
        shuffle_match = re.match(
            r'call\s+(\S+)\s+@__spirv_SubgroupShuffleXor\(i32\s+\d+,\s*(\S+)\s+(%\d+),\s*i32\s+(%\d+|\d+)\)',
            rhs
        )
        if shuffle_match:
            ret_type = shuffle_match.group(1)
            val = shuffle_match.group(3)
            mask = shuffle_match.group(4)
            wgsl_type = LLVM_TYPE_TO_WGSL.get(ret_type, 'i32')
            val_expr = self._operand(val, wgsl_type)
            mask_expr = self._operand(mask, 'u32')
            if not mask_expr.startswith('u32('):
                mask_expr = f'u32({mask_expr})'
            self.values[dst] = (self._var(dst), wgsl_type)

            if self.use_native_subgroups:
                # Native WGSL subgroup shuffle
                self.needs_subgroups = True
                return (f'let {self._var(dst)}: {wgsl_type} = '
                        f'subgroupShuffleXor({val_expr}, {mask_expr});')
            else:
                # Shared-memory emulation
                self.needs_shuffle_scratch = True
                return (f'_shfl[_lid.x] = {val_expr};\n'
                        f'    workgroupBarrier();\n'
                        f'    let {self._var(dst)}: {wgsl_type} = _shfl[_lid.x ^ {mask_expr}];\n'
                        f'    workgroupBarrier();')

        # --- atomicrmw instruction ---
        # Pattern: atomicrmw OP ptr addrspace(1) %ptr, TYPE %val ordering, align N
        atomic_match = re.match(
            r'atomicrmw\s+(\w+)\s+ptr\s+addrspace\(1\)\s+(%\d+),\s*(\w+)\s+(%\d+|[\d.e+-]+|0x[\da-fA-F]+)\s',
            rhs
        )
        if atomic_match:
            return self._translate_atomicrmw(
                dst, atomic_match.group(1), atomic_match.group(2),
                atomic_match.group(3), atomic_match.group(4))

        # --- insertvalue (struct aggregate) ---
        # Pattern: insertvalue { T1, T2, ... } %src_or_undef, T %val, IDX
        insertval_match = re.match(
            r'insertvalue\s+\{[^}]+\}\s+(%\d+|undef),\s*\w+\s+(%\d+|[\d.e+-]+|0x[\da-fA-F]+),\s*(\d+)',
            rhs
        )
        if insertval_match:
            src = insertval_match.group(1)
            val = insertval_match.group(2)
            idx = int(insertval_match.group(3))
            # Parse struct field types from the type signature
            type_match = re.match(r'insertvalue\s+\{([^}]+)\}', rhs)
            field_types_str = type_match.group(1) if type_match else 'float'
            field_types = [t.strip() for t in field_types_str.split(',')]
            wgsl_type = LLVM_TYPE_TO_WGSL.get(field_types[idx] if idx < len(field_types) else 'float', 'f32')

            # Copy fields from source (if not undef)
            fields = {}
            if src != 'undef' and src in self.struct_fields:
                fields = dict(self.struct_fields[src])

            val_expr = self._operand(val, wgsl_type)
            fields[idx] = val_expr

            self.struct_fields[dst] = fields
            # Also register individual field values for extractvalue
            for fi, fexpr in fields.items():
                ft = LLVM_TYPE_TO_WGSL.get(field_types[fi] if fi < len(field_types) else 'float', 'f32')
                self.values[f'{dst}_{fi}'] = (fexpr, ft)
            return None  # No statement — struct is virtual

        # --- extractvalue (struct aggregate) ---
        # Pattern: extractvalue { T1, T2, ... } %src, IDX
        extractval_match = re.match(
            r'extractvalue\s+\{([^}]+)\}\s+(%\d+),\s*(\d+)',
            rhs
        )
        if extractval_match:
            field_types_str = extractval_match.group(1)
            src = extractval_match.group(2)
            idx = int(extractval_match.group(3))
            field_types = [t.strip() for t in field_types_str.split(',')]
            wgsl_type = LLVM_TYPE_TO_WGSL.get(field_types[idx] if idx < len(field_types) else 'float', 'f32')

            if src in self.struct_fields and idx in self.struct_fields[src]:
                expr = self.struct_fields[src][idx]
            else:
                # Fallback: look for the field variable
                field_key = f'{src}_{idx}'
                if field_key in self.values:
                    expr = self.values[field_key][0]
                else:
                    expr = self._zero_val(wgsl_type)

            self.values[dst] = (expr, wgsl_type)
            return None

        # --- insertelement <N x T> (general, including <1 x T>) ---
        insert_match = re.match(
            r'insertelement\s+<(\d+)\s+x\s+(\w+)>\s+(%\d+|undef),\s*\w+\s+(%\d+|\d+|[\d.e+-]+|0x[\da-fA-F]+),\s*i32\s+(\d+)',
            rhs
        )
        if insert_match:
            vec_size = int(insert_match.group(1))
            elem_type = insert_match.group(2)
            src_vec = insert_match.group(3)
            val = insert_match.group(4)
            idx = int(insert_match.group(5))
            wgsl_type = LLVM_TYPE_TO_WGSL.get(elem_type, 'f32')
            val_expr = self._operand(val, wgsl_type)

            if vec_size == 1:
                # <1 x T>: pass-through
                self.values[dst] = (val_expr, wgsl_type)
                return None

            # <N x T>: track elements
            elements = {}
            if src_vec != 'undef' and src_vec in self.vector_elements:
                elements = dict(self.vector_elements[src_vec])
            elements[idx] = (val_expr, wgsl_type)
            self.vector_elements[dst] = elements
            return None

        # --- extractelement <N x T> (general, including <1 x T>) ---
        extract_match = re.match(
            r'extractelement\s+<(\d+)\s+x\s+(\w+)>\s+(%\d+),\s*i32\s+(\d+)',
            rhs
        )
        if extract_match:
            vec_size = int(extract_match.group(1))
            elem_type = extract_match.group(2)
            vec_val = extract_match.group(3)
            idx = int(extract_match.group(4))
            wgsl_type = LLVM_TYPE_TO_WGSL.get(elem_type, 'f32')

            if vec_size == 1:
                # <1 x T>: pass-through
                vec_expr = self._operand(vec_val, wgsl_type)
                self.values[dst] = (vec_expr, wgsl_type)
                return None

            # <N x T>: look up tracked element
            if vec_val in self.vector_elements and idx in self.vector_elements[vec_val]:
                expr, etype = self.vector_elements[vec_val][idx]
                self.values[dst] = (expr, etype)
            else:
                self.values[dst] = (self._zero_val(wgsl_type), wgsl_type)
            return None

        # --- Other call instructions (generic) ---
        call_match = re.match(r'call\s+(\S+)\s+@([\w.]+)\((.*)\)', rhs)
        if call_match:
            ret_type = call_match.group(1)
            func_name = call_match.group(2)
            args_str = call_match.group(3)

            # Try to handle known LLVM intrinsics
            intrinsic_result = self._translate_llvm_intrinsic(
                dst, ret_type, func_name, args_str)
            if intrinsic_result is not None:
                return intrinsic_result

            # Skip unknown SPIR-V/LLVM intrinsics with zero default
            wgsl_type = LLVM_TYPE_TO_WGSL.get(ret_type, 'i32')
            zv = self._zero_val(wgsl_type)
            self.values[dst] = (self._var(dst), wgsl_type)
            return f'let {self._var(dst)}: {wgsl_type} = {zv}; /* TODO: {func_name} */'


        # --- Cast instructions: fptosi, sitofp, zext, sext, trunc, etc. ---
        cast_match = re.match(
            r'(fptosi|sitofp|fptoui|uitofp|zext|sext|trunc|fpext|fptrunc|bitcast)'
            r'\s+(\S+)\s+(.+?)\s+to\s+(\S+)',
            rhs
        )
        if cast_match:
            cast_op = cast_match.group(1)
            src_type = cast_match.group(2)
            src_val = cast_match.group(3).strip()
            dst_type = cast_match.group(4)
            return self._translate_cast(dst, cast_op, src_type, src_val, dst_type)

        # --- GEP into shared memory (addrspace(3)) ---
        smem_gep_match = re.match(
            r'getelementptr\s+(?:inbounds\s+)?i8,\s*ptr\s+addrspace\(3\)\s+@global_smem,\s*i32\s+(.+)',
            rhs
        )
        if smem_gep_match:
            offset = smem_gep_match.group(1).strip()
            offset_expr = self._operand(offset, 'i32')
            self.smem_gep_info[dst] = offset_expr
            self.needs_smem = True
            return None

        # --- Nested GEP constant expression: base is a GEP into @global_smem with constant offset ---
        # Pattern: getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 BASE), i32 OFFSET
        nested_gep_match = re.match(
            r'getelementptr\s+(?:inbounds\s+)?i8,\s*ptr\s+addrspace\(3\)\s+'
            r'getelementptr\s*\(i8,\s*ptr\s+addrspace\(3\)\s+@global_smem,\s*i64\s+(\d+)\),\s*i32\s+(.+)',
            rhs
        )
        if nested_gep_match:
            base_offset = int(nested_gep_match.group(1))
            dyn_offset = nested_gep_match.group(2).strip()
            dyn_expr = self._operand(dyn_offset, 'i32')
            self.smem_gep_info[dst] = f'({base_offset} + {dyn_expr})'
            self.needs_smem = True
            return None

        # --- GEP with typed element into shared memory (e.g., float or i32) ---
        # Pattern: getelementptr inbounds float, ptr addrspace(3) %ptr, i32 N
        smem_typed_gep_match = re.match(
            r'getelementptr\s+(?:inbounds\s+)?(\w+),\s*ptr\s+addrspace\(3\)\s+(%\d+),\s*i32\s+(.+)',
            rhs
        )
        if smem_typed_gep_match:
            elem_type = smem_typed_gep_match.group(1)
            base_ptr = smem_typed_gep_match.group(2)
            idx = smem_typed_gep_match.group(3).strip()
            elem_bytes = LLVM_TYPE_TO_BYTES.get(elem_type, 4)
            idx_expr = self._operand(idx, 'i32')

            if base_ptr in self.smem_gep_info:
                base_expr = self.smem_gep_info[base_ptr]
                if elem_bytes == 1:
                    self.smem_gep_info[dst] = f'({base_expr} + {idx_expr})'
                else:
                    self.smem_gep_info[dst] = f'({base_expr} + ({idx_expr}) * {elem_bytes})'
                self.needs_smem = True
                return None

        # --- Chained GEP into shared memory with i8 base ---
        # Pattern: getelementptr i8, ptr addrspace(3) %known_smem_ptr, i32 OFFSET
        smem_chain_gep_match = re.match(
            r'getelementptr\s+(?:inbounds\s+)?i8,\s*ptr\s+addrspace\(3\)\s+(%\d+),\s*i32\s+(.+)',
            rhs
        )
        if smem_chain_gep_match:
            base_ptr = smem_chain_gep_match.group(1)
            offset = smem_chain_gep_match.group(2).strip()
            offset_expr = self._operand(offset, 'i32')
            if base_ptr in self.smem_gep_info:
                base_expr = self.smem_gep_info[base_ptr]
                self.smem_gep_info[dst] = f'({base_expr} + {offset_expr})'
                self.needs_smem = True
                return None

        # --- load from shared memory (addrspace(3)) ---
        smem_load_match = re.match(
            r'load\s+(\S+),\s*ptr\s+addrspace\(3\)\s+(%\d+)',
            rhs
        )
        if smem_load_match:
            elem_type = smem_load_match.group(1).rstrip(',')
            ptr = smem_load_match.group(2)
            return self._translate_smem_load(dst, ptr, elem_type)

        # --- GEP ---
        gep_match = re.match(
            r'getelementptr\s+(?:inbounds\s+)?(\S+),\s*ptr\s+addrspace\(1\)\s+(%\d+),\s*i32\s+(.+)',
            rhs
        )
        if gep_match:
            elem_type = gep_match.group(1).rstrip(',')
            base = gep_match.group(2)
            offset = gep_match.group(3).strip()
            return self._translate_gep(dst, base, offset, elem_type)

        # --- load ---
        load_match = re.match(
            r'load\s+(\S+),\s*ptr\s+addrspace\(1\)\s+(%\d+)',
            rhs
        )
        if load_match:
            elem_type = load_match.group(1).rstrip(',')
            ptr = load_match.group(2)
            return self._translate_load(dst, ptr, elem_type)

        # --- select ---
        select_match = re.match(
            r'select\s+i1\s+(%\d+),\s*(\S+)\s+(%\d+|[\d.e+-]+|undef),\s*\S+\s+(%\d+|[\d.e+-]+|undef)',
            rhs
        )
        if select_match:
            cond = select_match.group(1)
            ty = select_match.group(2).rstrip(',')
            true_val = select_match.group(3)
            false_val = select_match.group(4)
            return self._translate_select(dst, cond, true_val, false_val, ty)

        # --- Binary arithmetic/bitwise/comparison ops ---
        # Pattern: op [flags] type operand1, operand2
        binop_match = re.match(
            r'(\w+)\s*(?:(?:nsw|nuw|exact|disjoint)\s+)*(\S+)\s+(.+),\s*(.+)',
            rhs
        )
        if binop_match:
            op = binop_match.group(1)
            ty = binop_match.group(2)
            op1 = binop_match.group(3).strip()
            op2 = binop_match.group(4).strip()
            return self._translate_binop(dst, op, ty, op1, op2)

        # Unknown instruction
        return f'// UNKNOWN: {dst} = {rhs}'

    # -------------------------------------------------------------------
    # GEP handling
    # -------------------------------------------------------------------

    def _translate_gep(self, dst: str, base: str, offset: str,
                       elem_type: str) -> Optional[str]:
        """Translate a GEP instruction — doesn't emit code, just tracks provenance."""
        offset_expr = self._operand(offset, 'i32')
        offset_uniform = self._is_operand_uniform(offset)

        # Determine which buffer the base points to
        try:
            base_arg_idx = int(base[1:])  # %0, %1, etc.
            if base_arg_idx in self.arg_to_binding:
                self.gep_info[dst] = GepInfo(
                    buffer_arg_idx=base_arg_idx,
                    offset_expr=offset_expr,
                    is_uniform=offset_uniform,
                )
                return None  # No WGSL statement for GEP
        except ValueError:
            pass

        # If base is itself a GEP result (chained GEPs)
        if base in self.gep_info:
            parent = self.gep_info[base]
            # Combine parent offset with new offset
            combined_offset = f'({parent.offset_expr} + {offset_expr})'
            self.gep_info[dst] = GepInfo(
                buffer_arg_idx=parent.buffer_arg_idx,
                offset_expr=combined_offset,
                is_uniform=parent.is_uniform and offset_uniform,
            )
            return None

        return f'// GEP with unknown base: {base}'

    def _is_atomic_buffer(self, arg_idx: int) -> bool:
        """Check if a buffer argument has atomic operations."""
        return arg_idx in self.atomic_buffers

    def _is_operand_uniform(self, operand: str) -> bool:
        """Check if an operand (SSA name or literal) is thread-uniform.

        A value is uniform if it is the same across all threads in a workgroup:
        - Numeric literals / constants
        - WorkgroupId, NumWorkgroups builtins
        - Scalar function arguments (params)
        - Values derived purely from uniform operands
        """
        if operand in self.uniform_values:
            return True
        # Numeric literals are always uniform
        if operand.lstrip('-').replace('.', '', 1).replace('e', '', 1).replace('+', '', 1).isdigit():
            return True
        if operand.startswith('0x'):
            return True
        if operand in ('true', 'false', 'undef', 'zeroinitializer'):
            return True
        return False

    def _translate_load(self, dst: str, ptr: str, elem_type: str) -> str:
        """Translate a load instruction."""
        wgsl_type = LLVM_TYPE_TO_WGSL.get(elem_type, 'f32')

        if ptr in self.gep_info:
            gep = self.gep_info[ptr]
            binding_idx = self.arg_to_binding.get(gep.buffer_arg_idx, 0)
            buf_name = f'buf{binding_idx}'
            self.values[dst] = (self._var(dst), wgsl_type)
            if self._is_atomic_buffer(gep.buffer_arg_idx):
                # Atomic buffer: use atomicLoad; element type is atomic<i32>
                load_expr = f'atomicLoad(&{buf_name}[u32({gep.offset_expr})])'
                if wgsl_type == 'f32':
                    load_expr = f'bitcast<f32>({load_expr})'
                return f'let {self._var(dst)}: {wgsl_type} = {load_expr};'
            return f'let {self._var(dst)}: {wgsl_type} = {buf_name}[u32({gep.offset_expr})];'

        # Direct load from a function argument (no GEP → load at index 0)
        if ptr.startswith('%'):
            try:
                arg_idx = int(ptr[1:])
                if arg_idx in self.arg_to_binding:
                    binding_idx = self.arg_to_binding[arg_idx]
                    buf_name = f'buf{binding_idx}'
                    self.values[dst] = (self._var(dst), wgsl_type)
                    if self._is_atomic_buffer(arg_idx):
                        load_expr = f'atomicLoad(&{buf_name}[0u])'
                        if wgsl_type == 'f32':
                            load_expr = f'bitcast<f32>({load_expr})'
                        return f'let {self._var(dst)}: {wgsl_type} = {load_expr};'
                    return f'let {self._var(dst)}: {wgsl_type} = {buf_name}[0u];'
            except ValueError:
                pass

        self.values[dst] = (self._var(dst), wgsl_type)
        return f'let {self._var(dst)}: {wgsl_type} = {wgsl_type}(0); // load from unknown ptr {ptr}'

    def _translate_smem_load(self, dst: str, ptr: str, elem_type: str) -> str:
        """Translate a load from shared memory (addrspace(3))."""
        wgsl_type = LLVM_TYPE_TO_WGSL.get(elem_type, 'i32')
        if ptr in self.smem_gep_info:
            byte_offset = self.smem_gep_info[ptr]
            if wgsl_type == 'f16':
                # f16 is 2 bytes; shared memory is array<i32> (4 bytes per slot)
                # Pack two f16 values per i32 slot
                idx = f'u32({byte_offset}) >> 2u'
                shift = f'(u32({byte_offset}) & 2u) << 3u'  # 0 or 16
                expr = f'f16(bitcast<vec2<f16>>(u32(_smem[{idx}]))[u32({byte_offset}) >> 1u & 1u])'
                self.values[dst] = (self._var(dst), wgsl_type)
                return f'let {self._var(dst)}: {wgsl_type} = {expr};'
            idx = f'u32({byte_offset}) >> 2u'
            if wgsl_type == 'f32':
                expr = f'bitcast<f32>(_smem[{idx}])'
            else:
                expr = f'_smem[{idx}]'
            self.values[dst] = (self._var(dst), wgsl_type)
            return f'let {self._var(dst)}: {wgsl_type} = {expr};'
        self.values[dst] = (self._var(dst), wgsl_type)
        return f'let {self._var(dst)}: {wgsl_type} = {self._zero_val(wgsl_type)}; // smem load unknown ptr'

    def _translate_smem_store(self, val: str, ptr: str, ty: str) -> str:
        """Translate a store to shared memory (addrspace(3))."""
        wgsl_type = LLVM_TYPE_TO_WGSL.get(ty, 'i32')
        val_expr = self._operand(val, wgsl_type)
        if ptr in self.smem_gep_info:
            byte_offset = self.smem_gep_info[ptr]
            if wgsl_type == 'f16':
                # Store f16 into i32 shared memory: convert to i32 via vec2<f16>
                # Use the lower 16 bits of the i32 slot (simple approach: one f16 per i32 slot)
                idx = f'u32({byte_offset}) >> 2u'
                return f'_smem[{idx}] = bitcast<i32>(vec2<f16>({val_expr}, f16(0)));'
            idx = f'u32({byte_offset}) >> 2u'
            if wgsl_type == 'f32':
                return f'_smem[{idx}] = bitcast<i32>({val_expr});'
            else:
                return f'_smem[{idx}] = {val_expr};'
        return f'// smem store to unknown ptr {ptr}'

    def _translate_store(self, val: str, ptr: str, ty: str = 'float') -> str:
        """Translate a store instruction with bounds checking."""
        # Check if this is a masked store (value from select ... undef)
        mask_cond = self.masked_values.get(val)

        # Determine the buffer element type for proper casting
        buf_elem = 'f32'  # default
        binding_idx = 0
        if ptr in self.gep_info:
            gep = self.gep_info[ptr]
            binding_idx = self.arg_to_binding.get(gep.buffer_arg_idx, 0)
            if binding_idx < len(self.buffer_bindings):
                buf_elem = self.buffer_bindings[binding_idx].elem_type

        val_expr = self._operand(val, buf_elem)

        # Convert bool to numeric if storing to numeric buffer
        if val in self.values and self.values[val][1] == 'bool':
            if buf_elem == 'f32':
                val_expr = f'select(f32(0), f32(1), {self._operand(val, "bool")})'
            else:
                val_expr = f'select(0, 1, {self._operand(val, "bool")})'

        if ptr in self.gep_info:
            gep = self.gep_info[ptr]
            binding_idx = self.arg_to_binding.get(gep.buffer_arg_idx, 0)
            buf_name = f'buf{binding_idx}'
            idx_expr = gep.offset_expr

            # Atomic buffer: use atomicStore
            if self._is_atomic_buffer(gep.buffer_arg_idx):
                int_val = val_expr
                if buf_elem == 'f32':
                    int_val = f'bitcast<i32>({val_expr})'
                elif buf_elem != 'i32':
                    int_val = f'i32({val_expr})'
                if mask_cond:
                    cond_expr = self._operand(mask_cond, 'bool')
                    return (f'if {cond_expr} '
                            f'{{ atomicStore(&{buf_name}[u32({idx_expr})], {int_val}); }}')
                return (f'if u32({idx_expr}) < arrayLength(&{buf_name}) '
                        f'{{ atomicStore(&{buf_name}[u32({idx_expr})], {int_val}); }}')

            # For masked stores, use condition instead of bounds check
            if mask_cond:
                cond_expr = self._operand(mask_cond, 'bool')
                # Get the true value from the select (not the select result with zero fallback)
                # The true value is stored in the select var, and cond controls it
                return (f'if {cond_expr} '
                        f'{{ {buf_name}[u32({idx_expr})] = {val_expr}; }}')

            # Uniform offset: all threads compute the same index, so guard
            # with _lid.x == 0u to avoid D3D12 UAV concurrent-write issues
            if gep.is_uniform:
                return (f'if _lid.x == 0u '
                        f'{{ {buf_name}[u32({idx_expr})] = {val_expr}; }}')

            return (f'if u32({idx_expr}) < arrayLength(&{buf_name}) '
                    f'{{ {buf_name}[u32({idx_expr})] = {val_expr}; }}')

        # Direct store to a function argument (scalar store at element 0)
        # Guard with _lid.x == 0: all threads write to index 0 (same value),
        # but on some GPU backends (D3D12) concurrent writes to the same
        # storage element after workgroupBarrier() can be lost.
        if ptr.startswith('%'):
            try:
                arg_idx = int(ptr[1:])
                if arg_idx in self.arg_to_binding:
                    binding_idx = self.arg_to_binding[arg_idx]
                    buf_name = f'buf{binding_idx}'
                    if binding_idx < len(self.buffer_bindings):
                        buf_elem = self.buffer_bindings[binding_idx].elem_type
                        val_expr = self._operand(val, buf_elem)
                        # Convert bool to numeric if needed
                        if val in self.values and self.values[val][1] == 'bool':
                            if buf_elem == 'f32':
                                val_expr = f'select(f32(0), f32(1), {self._operand(val, "bool")})'
                            else:
                                val_expr = f'select(0, 1, {self._operand(val, "bool")})'
                    if mask_cond:
                        cond_expr = self._operand(mask_cond, 'bool')
                        return f'if {cond_expr} {{ {buf_name}[0u] = {val_expr}; }}'
                    return f'if _lid.x == 0u {{ {buf_name}[0u] = {val_expr}; }}'
            except ValueError:
                pass

        return f'// store to unknown ptr {ptr}'

    # -------------------------------------------------------------------
    # Atomic RMW instruction handling
    # -------------------------------------------------------------------

    def _translate_atomicrmw(self, dst: str, op: str, ptr: str,
                             val_type: str, val: str) -> str:
        """Translate an atomicrmw instruction to WGSL atomic builtins.

        LLVM: %old = atomicrmw OP ptr addrspace(1) %ptr, TYPE %val ...
        WGSL: atomicAdd(&buf_atomic[idx], val), etc.

        WGSL atomic builtins (i32/u32 only):
          atomicAdd, atomicSub, atomicMax, atomicMin,
          atomicAnd, atomicOr, atomicXor, atomicExchange,
          atomicCompareExchangeWeak, atomicLoad, atomicStore
        """
        is_float = val_type in ('float', 'half', 'double')
        wgsl_type = LLVM_TYPE_TO_WGSL.get(val_type, 'i32')
        val_expr = self._operand(val, wgsl_type)

        # Resolve the buffer and index from GEP info
        if ptr not in self.gep_info:
            self.values[dst] = (self._var(dst), wgsl_type)
            return f'let {self._var(dst)}: {wgsl_type} = {self._zero_val(wgsl_type)}; // atomicrmw unknown ptr'

        gep = self.gep_info[ptr]
        binding_idx = self.arg_to_binding.get(gep.buffer_arg_idx, 0)
        buf_name = f'buf{binding_idx}'
        idx_expr = gep.offset_expr

        # WGSL atomicrmw op mapping (integer ops → direct WGSL builtins)
        ATOMIC_OP_MAP = {
            'add': 'atomicAdd',
            'sub': 'atomicSub',
            'max': 'atomicMax',
            'min': 'atomicMin',
            'umax': 'atomicMax',  # unsigned max — use u32 cast
            'umin': 'atomicMin',  # unsigned min — use u32 cast
            'and': 'atomicAnd',
            'or': 'atomicOr',
            'xor': 'atomicXor',
            'xchg': 'atomicExchange',
        }

        if is_float and op == 'fadd':
            # Float atomic add: emulate with CAS loop
            # bitcast f32 ↔ i32 and use atomicCompareExchangeWeak
            self.values[dst] = (self._var(dst), wgsl_type)
            old_var = self._var(dst) + '_old'
            new_var = self._var(dst) + '_new'
            cas_var = self._var(dst) + '_cas'
            return (
                f'var {old_var}: i32 = atomicLoad(&{buf_name}[u32({idx_expr})]);\n'
                f'    loop {{\n'
                f'        let {new_var}: i32 = bitcast<i32>(bitcast<f32>({old_var}) + {val_expr});\n'
                f'        let {cas_var} = atomicCompareExchangeWeak(&{buf_name}[u32({idx_expr})], {old_var}, {new_var});\n'
                f'        if {cas_var}.exchanged {{ break; }}\n'
                f'        {old_var} = {cas_var}.old_value;\n'
                f'    }}\n'
                f'    let {self._var(dst)}: {wgsl_type} = bitcast<f32>({old_var});'
            )
        elif is_float and op in ('max', 'min', 'umax', 'umin'):
            # Float atomic max/min: emulate with CAS loop using integer comparison
            # Triton bitcasts floats to ints for these; use the integer path
            cmp_op = 'max' if op in ('max', 'umax') else 'min'
            wgsl_fn = f'atomicMax' if cmp_op == 'max' else f'atomicMin'
            self.values[dst] = (self._var(dst), 'i32')
            return (f'let {self._var(dst)}: i32 = '
                    f'{wgsl_fn}(&{buf_name}[u32({idx_expr})], {val_expr});')
        elif op in ATOMIC_OP_MAP:
            # Direct integer atomic operation
            wgsl_fn = ATOMIC_OP_MAP[op]
            self.values[dst] = (self._var(dst), 'i32')
            int_val = val_expr
            # Ensure the value is i32 for the atomic builtin
            if wgsl_type != 'i32':
                int_val = f'i32({val_expr})'
            return (f'let {self._var(dst)}: i32 = '
                    f'{wgsl_fn}(&{buf_name}[u32({idx_expr})], {int_val});')
        else:
            # Unknown atomic op — fallback
            self.values[dst] = (self._var(dst), 'i32')
            return f'let {self._var(dst)}: i32 = 0; // unsupported atomicrmw {op}'

    # -------------------------------------------------------------------
    # Cast instruction handling
    # -------------------------------------------------------------------

    def _translate_cast(self, dst: str, cast_op: str, src_type: str,
                        src_val: str, dst_type: str) -> str:
        """Translate LLVM cast instructions (fptosi, sitofp, zext, etc.)."""
        src_wgsl = LLVM_TYPE_TO_WGSL.get(src_type, 'i32')
        dst_wgsl = LLVM_TYPE_TO_WGSL.get(dst_type, 'i32')
        a = self._operand(src_val, src_wgsl)

        if cast_op == 'sitofp':
            float_ty = dst_wgsl if dst_wgsl in ('f32', 'f16') else 'f32'
            expr = f'{float_ty}({a})'
            wgsl_type = float_ty
        elif cast_op == 'fptosi':
            expr = f'i32({a})'
            wgsl_type = 'i32'
        elif cast_op == 'uitofp':
            float_ty = dst_wgsl if dst_wgsl in ('f32', 'f16') else 'f32'
            expr = f'{float_ty}(u32({a}))'
            wgsl_type = float_ty
        elif cast_op == 'fptoui':
            expr = f'u32({a})'
            wgsl_type = 'u32'
        elif cast_op == 'zext':
            # zext i1 → i32: convert bool to int
            if src_wgsl == 'bool' or src_type == 'i1':
                expr = f'select(0, 1, {self._operand(src_val, "bool")})'
            else:
                expr = f'i32(u32({a}))'
            wgsl_type = dst_wgsl if dst_wgsl != 'bool' else 'i32'
        elif cast_op == 'sext':
            expr = f'i32({a})'
            wgsl_type = dst_wgsl if dst_wgsl != 'bool' else 'i32'
        elif cast_op == 'trunc':
            expr = f'{dst_wgsl}({a})'
            wgsl_type = dst_wgsl if dst_wgsl != 'bool' else 'i32'
        elif cast_op in ('fpext', 'fptrunc'):
            expr = f'{dst_wgsl}({a})'
            wgsl_type = dst_wgsl
        elif cast_op == 'bitcast':
            if src_wgsl == 'f32' and dst_wgsl == 'i32':
                expr = f'bitcast<i32>({a})'
            elif src_wgsl == 'i32' and dst_wgsl == 'f32':
                expr = f'bitcast<f32>({a})'
            else:
                expr = a
            wgsl_type = dst_wgsl
        else:
            expr = f'0 /* unknown cast: {cast_op} */'
            wgsl_type = dst_wgsl

        self.values[dst] = (self._var(dst), wgsl_type)
        # Propagate uniformity through casts
        if self._is_operand_uniform(src_val):
            self.uniform_values.add(dst)
        return f'let {self._var(dst)}: {wgsl_type} = {expr};'

    # -------------------------------------------------------------------
    # LLVM intrinsic handling
    # -------------------------------------------------------------------

    def _translate_llvm_intrinsic(self, dst: str, ret_type: str,
                                  func_name: str, args_str: str) -> Optional[str]:
        """Translate known LLVM intrinsic calls to WGSL builtins.

        Returns the translated WGSL statement or None if the intrinsic
        is not recognized.
        """
        wgsl_type = LLVM_TYPE_TO_WGSL.get(ret_type, 'f32')

        # Parse arguments: "float %val" or "float %a, float %b"
        arg_vals = []
        if args_str.strip():
            arg_parts = [a.strip() for a in args_str.split(',')]
            for part in arg_parts:
                tokens = part.strip().split()
                if len(tokens) >= 2:
                    arg_vals.append(self._operand(tokens[-1], wgsl_type))
                elif len(tokens) == 1:
                    arg_vals.append(self._operand(tokens[0], wgsl_type))

        # Strip type suffix(es): llvm.fabs.f32 → llvm.fabs
        base_name = func_name
        # Try progressively stripping dot-separated suffix
        for _ in range(3):
            # Check unary
            for prefix, wgsl_fn in LLVM_UNARY_INTRINSICS.items():
                if base_name == prefix or base_name.startswith(prefix + '.'):
                    if arg_vals:
                        expr = f'{wgsl_fn}({arg_vals[0]})'
                        self.values[dst] = (self._var(dst), wgsl_type)
                        return f'let {self._var(dst)}: {wgsl_type} = {expr};'

            # Check binary
            for prefix, wgsl_fn in LLVM_BINARY_INTRINSICS.items():
                if base_name == prefix or base_name.startswith(prefix + '.'):
                    if len(arg_vals) >= 2:
                        expr = f'{wgsl_fn}({arg_vals[0]}, {arg_vals[1]})'
                        self.values[dst] = (self._var(dst), wgsl_type)
                        return f'let {self._var(dst)}: {wgsl_type} = {expr};'

            # Check ternary (fma, fmuladd)
            for prefix, wgsl_fn in LLVM_TERNARY_INTRINSICS.items():
                if base_name == prefix or base_name.startswith(prefix + '.'):
                    if len(arg_vals) >= 3:
                        expr = f'{wgsl_fn}({arg_vals[0]}, {arg_vals[1]}, {arg_vals[2]})'
                        self.values[dst] = (self._var(dst), wgsl_type)
                        return f'let {self._var(dst)}: {wgsl_type} = {expr};'

            # Strip one trailing suffix segment
            last_dot = base_name.rfind('.')
            if last_dot > 0:
                base_name = base_name[:last_dot]
            else:
                break

        return None  # Not a recognized intrinsic

    def _translate_select(self, dst: str, cond: str, true_val: str,
                          false_val: str, ty: str) -> str:
        """Translate a select instruction."""
        wgsl_type = LLVM_TYPE_TO_WGSL.get(ty, 'f32')
        cond_expr = self._operand(cond, 'bool')
        true_expr = self._operand(true_val, wgsl_type)
        false_expr = self._operand(false_val, wgsl_type)

        # Track masked select: select i1 %cond, T %val, T undef
        # This pattern means "val if cond, else don't care" — used for masked stores
        if false_val.strip() == 'undef':
            self.masked_values[dst] = cond

        # WGSL select(false_val, true_val, cond) — reversed from LLVM
        self.values[dst] = (self._var(dst), wgsl_type)
        # Propagate uniformity through select
        if (self._is_operand_uniform(cond) and
            self._is_operand_uniform(true_val) and
            self._is_operand_uniform(false_val)):
            self.uniform_values.add(dst)
        return f'let {self._var(dst)}: {wgsl_type} = select({false_expr}, {true_expr}, {cond_expr});'

    # -------------------------------------------------------------------
    # Binary operation handling
    # -------------------------------------------------------------------

    def _translate_binop(self, dst: str, op: str, ty: str,
                         op1: str, op2: str) -> Optional[str]:
        """Translate a binary operation."""
        wgsl_type = LLVM_TYPE_TO_WGSL.get(ty, 'i32')

        # Comparison operations: icmp, fcmp
        if op == 'icmp' or op == 'fcmp':
            return self._translate_cmp(dst, op, ty, op1, op2)

        # The type for the operands
        a = self._operand(op1, wgsl_type)
        b = self._operand(op2, wgsl_type)

        # Integer arithmetic
        if op == 'add':
            expr = f'{a} + {b}'
        elif op == 'sub':
            expr = f'{a} - {b}'
        elif op == 'mul':
            expr = f'{a} * {b}'
        elif op == 'sdiv':
            expr = f'{a} / {b}'
        elif op == 'srem':
            expr = f'{a} % {b}'
        # Unsigned arithmetic (need u32 operations)
        elif op == 'udiv':
            expr = f'i32(u32({a}) / u32({b}))'
        elif op == 'urem':
            expr = f'i32(u32({a}) % u32({b}))'
        # Bitwise
        elif op == 'and':
            expr = f'{a} & {b}'
        elif op == 'or':
            expr = f'{a} | {b}'
        elif op == 'xor':
            expr = f'{a} ^ {b}'
        # Shifts (shift amount must be u32 in WGSL)
        elif op == 'shl':
            expr = f'{a} << u32({b})'
        elif op == 'lshr':
            expr = f'i32(u32({a}) >> u32({b}))'
        elif op == 'ashr':
            expr = f'{a} >> u32({b})'
        # Float arithmetic
        elif op == 'fadd':
            expr = f'{a} + {b}'
        elif op == 'fsub':
            expr = f'{a} - {b}'
        elif op == 'fmul':
            expr = f'{a} * {b}'
        elif op == 'fdiv':
            expr = f'{a} / {b}'
        elif op == 'frem':
            expr = f'{a} % {b}'
        # Float unary (fneg is sometimes encoded as fsub 0, x)
        elif op == 'fneg':
            expr = f'-{a}'
        # Type conversions
        elif op == 'sitofp':
            expr = f'f32({a})'
            wgsl_type = 'f32'
        elif op == 'fptosi':
            expr = f'i32({a})'
            wgsl_type = 'i32'
        elif op == 'uitofp':
            expr = f'f32(u32({a}))'
            wgsl_type = 'f32'
        elif op == 'fptoui':
            expr = f'u32({a})'
            wgsl_type = 'u32'
        elif op == 'fpext':
            expr = f'f32({a})'
            wgsl_type = 'f32'
        elif op == 'fptrunc':
            expr = f'f16({a})'
            wgsl_type = 'f16'
        elif op == 'sext':
            expr = f'i32({a})'
            wgsl_type = 'i32'
        elif op == 'zext':
            expr = f'i32(u32({a}))'
            wgsl_type = 'i32'
        elif op == 'trunc':
            expr = f'i32({a})'
            wgsl_type = 'i32'
        elif op == 'bitcast':
            expr = a
        else:
            expr = f'0 /* unknown op: {op} */'

        self.values[dst] = (self._var(dst), wgsl_type)
        # Propagate uniformity: if both operands are uniform, result is uniform
        if self._is_operand_uniform(op1) and self._is_operand_uniform(op2):
            self.uniform_values.add(dst)
        return f'let {self._var(dst)}: {wgsl_type} = {expr};'

    def _translate_cmp(self, dst: str, op: str, pred_and_ty: str,
                       op1: str, op2: str) -> str:
        """Translate comparison instruction."""
        # For icmp: pred_and_ty is the predicate (e.g., 'slt'), op1 is type+val
        # Actually, our regex captured: op='icmp', ty=predicate, op1=type, op2=val1,val2
        # But the format is: icmp slt i32 %a, %b
        # After our regex: op='icmp', ty='slt', op1='i32', op2='%a, %b'
        # Wait, let me re-examine...

        # The regex matched: (\w+)\s*(flags)*(\S+)\s+(.+),\s*(.+)
        # For "icmp slt i32 %26, %3":
        #   op = 'icmp'
        #   ty = 'slt' (the predicate, captured as "type")
        #   op1 = 'i32 %26' (the actual type + first operand)
        #   op2 = '%3' (second operand)

        pred = pred_and_ty  # This is actually the predicate (slt, sge, etc.)

        # Parse op1 to extract the actual type and first operand
        parts = op1.strip().split()
        if len(parts) >= 2:
            actual_type = parts[0]
            first_operand = parts[1]
        else:
            actual_type = 'i32'
            first_operand = op1

        wgsl_type = LLVM_TYPE_TO_WGSL.get(actual_type, 'i32')
        a = self._operand(first_operand, wgsl_type)
        b = self._operand(op2.strip(), wgsl_type)

        # Signed comparisons
        cmp_ops = {
            'eq': '==', 'ne': '!=',
            'slt': '<', 'sle': '<=', 'sgt': '>', 'sge': '>=',
            'ult': '<', 'ule': '<=', 'ugt': '>', 'uge': '>=',
            # Float predicates
            'oeq': '==', 'one': '!=', 'ogt': '>', 'oge': '>=',
            'olt': '<', 'ole': '<=',
            'ueq': '==', 'une': '!=',
        }

        cmp_op = cmp_ops.get(pred, '==')

        # For unsigned comparisons, convert to u32
        if pred in ('ult', 'ule', 'ugt', 'uge') and wgsl_type == 'i32':
            a = f'u32({a})'
            b = f'u32({b})'

        self.values[dst] = (self._var(dst), 'bool')
        # Propagate uniformity through comparisons
        if self._is_operand_uniform(first_operand) and self._is_operand_uniform(op2.strip()):
            self.uniform_values.add(dst)
        return f'let {self._var(dst)}: bool = {a} {cmp_op} {b};'

    # -------------------------------------------------------------------
    # Helper: Convert LLVM operand to WGSL expression
    # -------------------------------------------------------------------

    def _operand(self, op: str, expected_type: str = 'i32') -> str:
        """Convert an LLVM operand to a WGSL expression."""
        op = op.strip()

        # undef / zeroinitializer → zero value
        if op == 'undef' or op == 'zeroinitializer':
            return self._zero_val(expected_type)

        # SSA value reference
        if op.startswith('%'):
            # Check if it's a function argument
            try:
                arg_idx = int(op[1:])
                if arg_idx < len(self.func_args):
                    # Is this a scalar arg?
                    if arg_idx in self.arg_to_param:
                        param_name = self.arg_to_param[arg_idx]
                        return f'params.{param_name}'
                    # Is this a pointer arg? (shouldn't be used directly as a value)
                    if arg_idx in self.arg_to_binding:
                        return f'0 /* ptr arg {arg_idx} */'
            except ValueError:
                pass

            # Previously computed SSA value
            if op in self.values:
                return self.values[op][0]
            return self._var(op)

        # Integer literal
        if re.match(r'^-?\d+$', op):
            val = int(op)
            if expected_type in ('f32', 'f16'):
                return f'{expected_type}({val})'
            if expected_type == 'bool':
                return 'true' if val != 0 else 'false'
            return str(val)

        # Float literal
        if re.match(r'^-?[\d.]+(?:e[+-]?\d+)?$', op):
            val = float(op)
            if expected_type in ('f32', 'f16'):
                return f'{expected_type}({val})'
            return str(val)

        # Hex literal — convert LLVM hex double to float value
        if op.startswith('0x'):
            try:
                # LLVM represents float constants as the hex encoding
                # of a double-precision value
                int_val = int(op, 16)
                double_val = struct.unpack('d', struct.pack('Q', int_val))[0]
                if expected_type in ('f32', 'f16'):
                    return f'{expected_type}({double_val})'
                return str(double_val)
            except (ValueError, struct.error):
                return op

        return f'0 /* unknown operand: {op} */'

    def _var(self, ssa_name: str) -> str:
        """Convert SSA name like %7 to WGSL variable name like v7."""
        if ssa_name.startswith('%'):
            return f'v{ssa_name[1:]}'
        return f'v_{ssa_name}'

    def _zero_val(self, wgsl_type: str) -> str:
        """Return the zero value for a WGSL type."""
        if wgsl_type == 'f32':
            return 'f32(0)'
        if wgsl_type == 'f16':
            return 'f16(0)'
        if wgsl_type == 'bool':
            return 'false'
        if wgsl_type == 'u32':
            return 'u32(0)'
        return '0'

    # -------------------------------------------------------------------
    # Step 8: Emit WGSL code
    # -------------------------------------------------------------------

    def _emit_wgsl(self, stmts: List[str]) -> str:
        """Generate the complete WGSL shader code."""
        lines = []

        # Enable extensions if needed
        if self.needs_f16:
            lines.append('enable f16;')
        if self.needs_subgroups:
            lines.append('enable subgroups;')
        if self.needs_f16 or self.needs_subgroups:
            lines.append('')

        lines.append(f'// Auto-generated by Triton WebGPU Backend')
        lines.append(f'// Kernel: {self.kernel_name}')
        lines.append(f'// Workgroup size: {self.workgroup_size}'
                     f' ({self.num_warps} warps x {self.warp_size} threads)')
        lines.append('')

        # Shuffle scratch buffer for SubgroupShuffleXor emulation
        if self.needs_shuffle_scratch:
            lines.append(f'var<workgroup> _shfl: array<i32, {self.workgroup_size}>;')
            lines.append('')

        # Shared memory for cross-warp communication (addrspace(3))
        if self.needs_smem:
            smem_i32_count = max(self.num_warps, (self.smem_bytes + 3) // 4)
            lines.append(f'var<workgroup> _smem: array<i32, {smem_i32_count}>;')
            lines.append('')

        # Buffer declarations
        # Build reverse map: binding_idx → func_arg_idx for atomic detection
        binding_to_arg = {}
        for arg_idx, bind_idx in self.arg_to_binding.items():
            binding_to_arg[bind_idx] = arg_idx

        for bb in self.buffer_bindings:
            access = 'read_write' if bb.access == 'read_write' else 'read'
            arg_idx = binding_to_arg.get(bb.binding, -1)
            is_atomic = arg_idx in self.atomic_buffers
            if is_atomic:
                # Atomic buffers use atomic<i32> element type
                lines.append(f'@group(0) @binding({bb.binding}) '
                            f'var<storage, {access}> buf{bb.binding}: '
                            f'array<atomic<i32>>;  // {bb.name} (atomic)')
            else:
                lines.append(f'@group(0) @binding({bb.binding}) '
                            f'var<storage, {access}> buf{bb.binding}: '
                            f'array<{bb.elem_type}>;  // {bb.name}')
        lines.append('')

        # Params struct (scalar arguments)
        if self.param_fields:
            lines.append('struct Params {')
            for pf in self.param_fields:
                lines.append(f'    {pf.name}: {pf.wgsl_type},')
            lines.append('};')
            param_binding = len(self.buffer_bindings)
            lines.append(f'@group(0) @binding({param_binding}) '
                        f'var<storage, read> params: Params;')
            lines.append('')

        # Compute entry point
        lines.append(f'@compute @workgroup_size({self.workgroup_size})')
        lines.append('fn main(')
        lines.append('    @builtin(workgroup_id) _wg_id: vec3<u32>,')
        lines.append('    @builtin(local_invocation_id) _lid: vec3<u32>,')
        lines.append('    @builtin(num_workgroups) _num_wg: vec3<u32>,')
        lines.append(') {')

        # Body
        for stmt in stmts:
            if stmt:
                lines.append(f'    {stmt}')

        lines.append('}')
        lines.append('')

        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_llvm_to_wgsl(llir: str, signature: dict,
                           num_warps: int = 4,
                           warp_size: int = 32,
                           use_native_subgroups: bool = False) -> TranslationResult:
    """
    Translate LLVM IR from Triton's WebGPU backend to WGSL.

    Args:
        llir: LLVM IR string
        signature: Triton parameter signature dict (excluding constexprs)
        num_warps: Warps per workgroup
        warp_size: Threads per warp
        use_native_subgroups: Use native subgroupShuffleXor if adapter supports it

    Returns:
        TranslationResult with WGSL code, bindings, and metadata
    """
    translator = LLVMToWGSL(llir, signature, num_warps, warp_size,
                             use_native_subgroups=use_native_subgroups)
    return translator.translate()
