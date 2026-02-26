"""
WebGPU Runtime Kernel Runner
==============================

Executes WGSL compute shaders on the GPU via wgpu-py (wgpu-native/Vulkan).

Usage:
    runner = WebGPURunner()
    result = runner.run_kernel(
        wgsl_code=wgsl,
        buffer_bindings=[...],
        param_fields=[...],
        workgroup_size=128,
        grid=(num_workgroups,),
        buffers={'x_ptr': x_np, 'y_ptr': y_np, 'output_ptr': output_np},
        scalars={'n_elements': n},
    )
    output = result['output_ptr']  # numpy array
"""

import struct
import numpy as np

try:
    import wgpu
    import wgpu.utils
    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False

from .llvm_to_wgsl import BufferBinding, ParamField


WGSL_TYPE_TO_NUMPY = {
    'f32': np.float32,
    'f16': np.float16,
    'i32': np.int32,
    'u32': np.uint32,
}

WGSL_TYPE_TO_STRUCT_FMT = {
    'f32': '<f',
    'i32': '<i',
    'u32': '<I',
}


class WebGPURunner:
    """Execute WGSL compute shaders on the GPU via wgpu-py."""

    def __init__(self, adapter=None, device=None):
        if not HAS_WGPU:
            raise RuntimeError("wgpu-py is not installed. Run: pip install wgpu")

        if device is not None:
            self._adapter = adapter
            self._device = device
        else:
            self._adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance"
            )
            self._device = self._adapter.request_device_sync(
                required_limits=self._get_limits()
            )

    def _get_limits(self) -> dict:
        """Request generous device limits for compute."""
        return {
            "max-bind-groups": 4,
            "max-storage-buffers-per-shader-stage": 16,
            "max-storage-buffer-binding-size": 1 << 30,  # 1 GiB
            "max-buffer-size": 1 << 30,
            "max-compute-workgroups-per-dimension": 65535,
            "max-compute-invocations-per-workgroup": 256,
            "max-compute-workgroup-size-x": 256,
        }

    @property
    def adapter_info(self) -> str:
        """Return GPU adapter description."""
        info = self._adapter.info
        return f"{info.get('device', 'unknown')} ({info.get('backend_type', '?')})"

    def run_kernel(
        self,
        wgsl_code: str,
        buffer_bindings: list,
        param_fields: list,
        workgroup_size: int,
        grid: tuple,
        buffers: dict,
        scalars: dict = None,
    ) -> dict:
        """
        Execute a WGSL compute shader.

        Args:
            wgsl_code: Complete WGSL shader source
            buffer_bindings: List of BufferBinding (from translator)
            param_fields: List of ParamField (from translator)
            workgroup_size: Threads per workgroup
            grid: Tuple of (num_workgroups_x, [y, [z]])
            buffers: Dict mapping buffer name → numpy array (input/output data)
            scalars: Dict mapping scalar param name → int/float value

        Returns:
            Dict mapping output buffer names → numpy arrays with results
        """
        scalars = scalars or {}
        device = self._device

        # 1. Create the shader module
        shader = device.create_shader_module(code=wgsl_code)

        # 2. Create GPU buffers for each binding
        gpu_buffers = {}
        binding_entries = []

        for bb in buffer_bindings:
            if bb.name in buffers:
                # User-provided buffer
                np_arr = np.ascontiguousarray(buffers[bb.name])
                buf_size = np_arr.nbytes
                usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
                if bb.access == 'read':
                    usage |= wgpu.BufferUsage.COPY_DST
                else:
                    usage |= wgpu.BufferUsage.COPY_DST

                gpu_buf = device.create_buffer(size=buf_size, usage=usage)
                device.queue.write_buffer(gpu_buf, 0, np_arr.tobytes())
            else:
                # Internal/unused buffer — create a dummy 16-byte buffer
                buf_size = 16
                usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
                gpu_buf = device.create_buffer(size=buf_size, usage=usage)

            gpu_buffers[bb.name] = gpu_buf

            binding_entries.append({
                "binding": bb.binding,
                "resource": {
                    "buffer": gpu_buf,
                    "offset": 0,
                    "size": gpu_buf.size,
                },
            })

        # 3. Create params buffer (scalar arguments)
        if param_fields:
            params_data = bytearray()
            for pf in param_fields:
                val = scalars.get(pf.name, 0)
                fmt = WGSL_TYPE_TO_STRUCT_FMT.get(pf.wgsl_type, '<i')
                params_data.extend(struct.pack(fmt, val))

            # Pad to minimum buffer size (16 bytes)
            while len(params_data) < 16:
                params_data.extend(b'\x00')

            params_binding = len(buffer_bindings)
            params_buf = device.create_buffer(
                size=len(params_data),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            device.queue.write_buffer(params_buf, 0, bytes(params_data))

            binding_entries.append({
                "binding": params_binding,
                "resource": {
                    "buffer": params_buf,
                    "offset": 0,
                    "size": params_buf.size,
                },
            })

        # 4. Create bind group layout and bind group
        bind_group_layout_entries = []
        for i, entry in enumerate(binding_entries):
            is_params = (i == len(buffer_bindings) and param_fields)
            if is_params:
                buf_type = "read-only-storage"
            else:
                bb = buffer_bindings[i] if i < len(buffer_bindings) else None
                if bb and bb.access == 'read':
                    buf_type = "read-only-storage"
                else:
                    buf_type = "storage"

            bind_group_layout_entries.append({
                "binding": entry["binding"],
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": buf_type},
            })

        bind_group_layout = device.create_bind_group_layout(
            entries=bind_group_layout_entries
        )

        bind_group = device.create_bind_group(
            layout=bind_group_layout,
            entries=binding_entries,
        )

        # 5. Create compute pipeline
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "main"},
        )

        # 6. Encode and submit commands
        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)

        # Grid dimensions
        gx = grid[0] if len(grid) > 0 else 1
        gy = grid[1] if len(grid) > 1 else 1
        gz = grid[2] if len(grid) > 2 else 1
        compute_pass.dispatch_workgroups(gx, gy, gz)
        compute_pass.end()

        device.queue.submit([command_encoder.finish()])

        # 7. Read back output buffers
        results = {}
        for bb in buffer_bindings:
            if bb.name in buffers and bb.access == 'read_write':
                gpu_buf = gpu_buffers[bb.name]
                raw = device.queue.read_buffer(gpu_buf)
                np_dtype = WGSL_TYPE_TO_NUMPY.get(bb.elem_type, np.float32)
                results[bb.name] = np.frombuffer(raw, dtype=np_dtype).copy()

        return results


def run_triton_kernel_on_webgpu(
    compiled_kernel,
    grid: tuple,
    **kwargs,
) -> dict:
    """
    High-level API: Execute a compiled Triton WebGPU kernel on the GPU.

    Args:
        compiled_kernel: Result of triton.compile() for WebGPU target
        grid: Tuple of workgroup counts (x,) or (x, y) or (x, y, z)
        **kwargs: Named arguments matching the kernel signature.
                  Pointer args should be numpy arrays.
                  Scalar args should be int/float.

    Returns:
        Dict mapping output buffer names → numpy arrays with GPU results
    """
    from .llvm_to_wgsl import translate_llvm_to_wgsl

    # Get the LLVM IR from compilation
    llir = compiled_kernel.asm.get('llir', '')
    if not llir:
        raise ValueError("Compiled kernel has no LLVM IR. Was it compiled for WebGPU?")

    # Get metadata
    metadata = compiled_kernel.metadata
    sig = {}
    # Build signature from metadata
    # The signature info may be in different places depending on Triton version
    if hasattr(compiled_kernel, 'signature'):
        sig = compiled_kernel.signature

    num_warps = metadata.get('num_warps', 4)
    warp_size = 32

    # Translate to WGSL
    result = translate_llvm_to_wgsl(llir, sig, num_warps, warp_size)

    # Separate buffer vs scalar args
    buffers = {}
    scalars = {}
    for name, val in kwargs.items():
        if isinstance(val, np.ndarray):
            buffers[name] = val
        else:
            scalars[name] = val

    # Run
    runner = WebGPURunner()
    return runner.run_kernel(
        wgsl_code=result.wgsl,
        buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields,
        workgroup_size=result.workgroup_size,
        grid=grid,
        buffers=buffers,
        scalars=scalars,
    )
