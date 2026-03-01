"""
Dawn WebGPU Runtime Kernel Runner
===================================

Executes WGSL compute shaders on the GPU via Dawn's native WebGPU C API.

Dawn natively supports D3D12 (Windows), Vulkan (Linux), and Metal (macOS),
consuming WGSL shaders directly through its Tint compiler.

Usage:
    runner = DawnRunner()
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

import ctypes
import ctypes.util
import struct
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from .llvm_to_wgsl import BufferBinding, ParamField

# ============================================================================
# Dawn library discovery
# ============================================================================

_dawn_lib = None
_dawn_loaded = False

def _find_dawn_dll():
    """Locate the Dawn webgpu_dawn shared library."""
    # 1. Check DAWN_PATH environment variable
    dawn_path = os.environ.get("DAWN_PATH")
    if dawn_path and os.path.exists(dawn_path):
        return dawn_path

    # 2. Check relative to this file (build output locations)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    # In-tree build location
    triton_root = os.path.normpath(os.path.join(here, "..", "..", "..", ".."))
    candidates.extend([
        os.path.join(triton_root, "third_party", "webgpu", "dawn", "build", "webgpu_dawn.dll"),
        os.path.join(triton_root, "third_party", "webgpu", "dawn", "build", "libwebgpu_dawn.so"),
        os.path.join(triton_root, "third_party", "webgpu", "dawn", "build", "libwebgpu_dawn.dylib"),
    ])

    # Backend lib directory
    candidates.extend([
        os.path.join(here, "lib", "webgpu_dawn.dll"),
        os.path.join(here, "lib", "libwebgpu_dawn.so"),
        os.path.join(here, "lib", "libwebgpu_dawn.dylib"),
    ])

    for path in candidates:
        if os.path.exists(path):
            return path

    # 3. System search
    lib = ctypes.util.find_library("webgpu_dawn")
    if lib:
        return lib

    return None


def _load_dawn():
    """Load the Dawn library and set up function prototypes."""
    global _dawn_lib, _dawn_loaded
    if _dawn_loaded:
        return _dawn_lib

    path = _find_dawn_dll()
    if path is None:
        _dawn_loaded = True
        return None

    try:
        # Dawn uses LoadLibraryExA with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR which
        # requires absolute paths. Pre-load system DLLs that Dawn will need
        # (d3dcompiler_47.dll, vulkan-1.dll) so Dawn finds them already loaded.
        if sys.platform == 'win32':
            _preload_system_dlls()

        _dawn_lib = ctypes.CDLL(path)
    except OSError:
        _dawn_loaded = True
        return None

    _setup_prototypes(_dawn_lib)
    _dawn_loaded = True
    return _dawn_lib


def _preload_system_dlls():
    """Pre-load system DLLs that Dawn depends on (Windows only).

    Dawn's DynamicLib::Open uses LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR which
    fails with ERROR_INVALID_PARAMETER for relative filenames. Pre-loading
    the DLLs with standard LoadLibrary makes them available via
    GetModuleHandle when Dawn calls OpenLoaded.
    """
    kernel32 = ctypes.windll.kernel32

    # d3dcompiler_47.dll — needed for D3D11/D3D12 shader compilation
    try:
        kernel32.LoadLibraryW("d3dcompiler_47.dll")
    except Exception:
        pass

    # dxgi.dll — needed for D3D adapter enumeration
    try:
        kernel32.LoadLibraryW("dxgi.dll")
    except Exception:
        pass

    # vulkan-1.dll — needed for Vulkan backend
    vulkan_paths = []
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        vulkan_paths.append(os.path.join(vulkan_sdk, "Bin", "vulkan-1.dll"))
    vulkan_paths.append("vulkan-1.dll")  # system default

    for vp in vulkan_paths:
        try:
            kernel32.LoadLibraryW(vp)
            break
        except Exception:
            pass


def HAS_DAWN():
    """Check if Dawn is available."""
    return _load_dawn() is not None


# ============================================================================
# ctypes struct definitions matching Dawn's webgpu.h
# ============================================================================

# Opaque handle types — all are pointers to opaque structs
WGPUInstance = ctypes.c_void_p
WGPUAdapter = ctypes.c_void_p
WGPUDevice = ctypes.c_void_p
WGPUQueue = ctypes.c_void_p
WGPUShaderModule = ctypes.c_void_p
WGPUComputePipeline = ctypes.c_void_p
WGPUBuffer = ctypes.c_void_p
WGPUBindGroup = ctypes.c_void_p
WGPUBindGroupLayout = ctypes.c_void_p
WGPUPipelineLayout = ctypes.c_void_p
WGPUCommandEncoder = ctypes.c_void_p
WGPUCommandBuffer = ctypes.c_void_p
WGPUComputePassEncoder = ctypes.c_void_p
WGPUSurface = ctypes.c_void_p
WGPUSampler = ctypes.c_void_p
WGPUTextureView = ctypes.c_void_p

# Basic types
WGPUBool = ctypes.c_uint32
WGPUFlags = ctypes.c_uint64
WGPUBufferUsageFlags = WGPUFlags
WGPUMapModeFlags = WGPUFlags
WGPUShaderStageFlags = WGPUFlags

# size_t — platform dependent
SIZE_T = ctypes.c_size_t
WGPU_STRLEN = SIZE_T(-1).value  # SIZE_MAX sentinel for null-terminated strings


# ── Enums ──

class WGPUSType:
    ShaderSourceSPIRV = 0x00000001
    ShaderSourceWGSL = 0x00000002
    DawnTogglesDescriptor = 0x0005000A


class WGPUCallbackMode:
    WaitAnyOnly = 0x00000001
    AllowProcessEvents = 0x00000002
    AllowSpontaneous = 0x00000003


class WGPUWaitStatus:
    Success = 0x00000001
    TimedOut = 0x00000002
    Error = 0x00000003


class WGPUMapAsyncStatus:
    Success = 0x00000001
    CallbackCancelled = 0x00000002
    Error = 0x00000003
    Aborted = 0x00000004


class WGPURequestAdapterStatus:
    Success = 0x00000001


class WGPURequestDeviceStatus:
    Success = 0x00000001


class WGPUCreatePipelineAsyncStatus:
    Success = 0x00000001


class WGPUBufferBindingType:
    BindingNotUsed = 0x00000000
    Undefined = 0x00000001
    Uniform = 0x00000002
    Storage = 0x00000003
    ReadOnlyStorage = 0x00000004


class WGPUBackendType:
    Undefined = 0x00000000
    Null = 0x00000001
    WebGPU = 0x00000002
    D3D11 = 0x00000003
    D3D12 = 0x00000004
    Metal = 0x00000005
    Vulkan = 0x00000006


class WGPUFeatureLevel:
    Undefined = 0x00000000
    Compatibility = 0x00000001
    Core = 0x00000002


class WGPUPowerPreference:
    Undefined = 0x00000000
    LowPower = 0x00000001
    HighPerformance = 0x00000002


# Buffer usage flags
BUFFER_USAGE_MAP_READ = 0x0001
BUFFER_USAGE_MAP_WRITE = 0x0002
BUFFER_USAGE_COPY_SRC = 0x0004
BUFFER_USAGE_COPY_DST = 0x0008
BUFFER_USAGE_STORAGE = 0x0080

# Map mode flags
MAP_MODE_READ = 0x0001

# Shader stage flags
SHADER_STAGE_COMPUTE = 0x0004


# ── Structs ──

class WGPUStringView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_char_p),
        ("length", SIZE_T),
    ]

    @staticmethod
    def from_str(s):
        """Create a WGPUStringView from a Python string."""
        if s is None:
            return WGPUStringView(None, 0)
        encoded = s.encode("utf-8")
        return WGPUStringView(encoded, len(encoded))

    @staticmethod
    def null_terminated(s):
        """Create a WGPUStringView with WGPU_STRLEN sentinel."""
        encoded = s.encode("utf-8")
        return WGPUStringView(encoded, WGPU_STRLEN)


class WGPUChainedStruct(ctypes.Structure):
    pass

WGPUChainedStruct._fields_ = [
    ("next", ctypes.POINTER(WGPUChainedStruct)),
    ("sType", ctypes.c_uint32),
]


class WGPUDawnTogglesDescriptor(ctypes.Structure):
    """Dawn-specific toggle descriptor, chainable to adapter/device descriptors."""
    _fields_ = [
        ("chain", WGPUChainedStruct),
        ("enabledToggleCount", SIZE_T),
        ("enabledToggles", ctypes.POINTER(ctypes.c_char_p)),
        ("disabledToggleCount", SIZE_T),
        ("disabledToggles", ctypes.POINTER(ctypes.c_char_p)),
    ]


class WGPUFuture(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64)]


class WGPUFutureWaitInfo(ctypes.Structure):
    _fields_ = [
        ("future", WGPUFuture),
        ("completed", WGPUBool),
    ]


# ── Callback types ──
# void (*)(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* ud1, void* ud2)
RequestAdapterCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, WGPUAdapter, WGPUStringView, ctypes.c_void_p, ctypes.c_void_p
)

# void (*)(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* ud1, void* ud2)
RequestDeviceCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, WGPUDevice, WGPUStringView, ctypes.c_void_p, ctypes.c_void_p
)

# void (*)(WGPUMapAsyncStatus status, WGPUStringView message, void* ud1, void* ud2)
BufferMapCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, WGPUStringView, ctypes.c_void_p, ctypes.c_void_p
)

# void (*)(const WGPUDevice*, WGPUErrorType type, WGPUStringView message, void* ud1, void* ud2)
UncapturedErrorCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_uint32, WGPUStringView, ctypes.c_void_p, ctypes.c_void_p
)

# void (*)(WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline pipeline,
#          WGPUStringView message, void* ud1, void* ud2)
CreateComputePipelineAsyncCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, WGPUComputePipeline, WGPUStringView,
    ctypes.c_void_p, ctypes.c_void_p
)


# ── Callback info structs ──

class WGPURequestAdapterCallbackInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("mode", ctypes.c_uint32),
        ("callback", RequestAdapterCallback),
        ("userdata1", ctypes.c_void_p),
        ("userdata2", ctypes.c_void_p),
    ]


class WGPURequestDeviceCallbackInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("mode", ctypes.c_uint32),
        ("callback", RequestDeviceCallback),
        ("userdata1", ctypes.c_void_p),
        ("userdata2", ctypes.c_void_p),
    ]


class WGPUBufferMapCallbackInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("mode", ctypes.c_uint32),
        ("callback", BufferMapCallback),
        ("userdata1", ctypes.c_void_p),
        ("userdata2", ctypes.c_void_p),
    ]


class WGPUCreateComputePipelineAsyncCallbackInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("mode", ctypes.c_uint32),
        ("callback", CreateComputePipelineAsyncCallback),
        ("userdata1", ctypes.c_void_p),
        ("userdata2", ctypes.c_void_p),
    ]


# ── Descriptor structs ──

class WGPUInstanceDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("requiredFeatureCount", SIZE_T),
        ("requiredFeatures", ctypes.c_void_p),
        ("requiredLimits", ctypes.c_void_p),
    ]


class WGPURequestAdapterOptions(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("featureLevel", ctypes.c_uint32),
        ("powerPreference", ctypes.c_uint32),
        ("forceFallbackAdapter", WGPUBool),
        ("backendType", ctypes.c_uint32),
        ("compatibleSurface", WGPUSurface),
    ]


class WGPUAdapterInfo(ctypes.Structure):
    """Adapter information: vendor, architecture, device, description, backend."""
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("vendor", WGPUStringView),
        ("architecture", WGPUStringView),
        ("device", WGPUStringView),
        ("description", WGPUStringView),
        ("backendType", ctypes.c_uint32),
        ("adapterType", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
    ]


class WGPUQueueDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
    ]


class WGPUDeviceLostCallbackInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("mode", ctypes.c_uint32),
        ("callback", ctypes.c_void_p),
        ("userdata1", ctypes.c_void_p),
        ("userdata2", ctypes.c_void_p),
    ]


class WGPUUncapturedErrorCallbackInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("callback", ctypes.c_void_p),
        ("userdata1", ctypes.c_void_p),
        ("userdata2", ctypes.c_void_p),
    ]


class WGPULimits(ctypes.Structure):
    """WebGPU device/adapter limits (matches dawn/webgpu.h WGPULimits)."""
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("maxTextureDimension1D", ctypes.c_uint32),
        ("maxTextureDimension2D", ctypes.c_uint32),
        ("maxTextureDimension3D", ctypes.c_uint32),
        ("maxTextureArrayLayers", ctypes.c_uint32),
        ("maxBindGroups", ctypes.c_uint32),
        ("maxBindGroupsPlusVertexBuffers", ctypes.c_uint32),
        ("maxBindingsPerBindGroup", ctypes.c_uint32),
        ("maxDynamicUniformBuffersPerPipelineLayout", ctypes.c_uint32),
        ("maxDynamicStorageBuffersPerPipelineLayout", ctypes.c_uint32),
        ("maxSampledTexturesPerShaderStage", ctypes.c_uint32),
        ("maxSamplersPerShaderStage", ctypes.c_uint32),
        ("maxStorageBuffersPerShaderStage", ctypes.c_uint32),
        ("maxStorageTexturesPerShaderStage", ctypes.c_uint32),
        ("maxUniformBuffersPerShaderStage", ctypes.c_uint32),
        ("maxUniformBufferBindingSize", ctypes.c_uint64),
        ("maxStorageBufferBindingSize", ctypes.c_uint64),
        ("minUniformBufferOffsetAlignment", ctypes.c_uint32),
        ("minStorageBufferOffsetAlignment", ctypes.c_uint32),
        ("maxVertexBuffers", ctypes.c_uint32),
        ("maxBufferSize", ctypes.c_uint64),
        ("maxVertexAttributes", ctypes.c_uint32),
        ("maxVertexBufferArrayStride", ctypes.c_uint32),
        ("maxInterStageShaderVariables", ctypes.c_uint32),
        ("maxColorAttachments", ctypes.c_uint32),
        ("maxColorAttachmentBytesPerSample", ctypes.c_uint32),
        ("maxComputeWorkgroupStorageSize", ctypes.c_uint32),
        ("maxComputeInvocationsPerWorkgroup", ctypes.c_uint32),
        ("maxComputeWorkgroupSizeX", ctypes.c_uint32),
        ("maxComputeWorkgroupSizeY", ctypes.c_uint32),
        ("maxComputeWorkgroupSizeZ", ctypes.c_uint32),
        ("maxComputeWorkgroupsPerDimension", ctypes.c_uint32),
        ("maxImmediateSize", ctypes.c_uint32),
    ]


class WGPUDeviceDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("requiredFeatureCount", SIZE_T),
        ("requiredFeatures", ctypes.c_void_p),
        ("requiredLimits", ctypes.POINTER(WGPULimits)),
        ("defaultQueue", WGPUQueueDescriptor),
        ("deviceLostCallbackInfo", WGPUDeviceLostCallbackInfo),
        ("uncapturedErrorCallbackInfo", WGPUUncapturedErrorCallbackInfo),
    ]


class WGPUShaderModuleDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
    ]


class WGPUShaderSourceWGSL(ctypes.Structure):
    _fields_ = [
        ("chain", WGPUChainedStruct),  # embedded by value, not pointer
        ("code", WGPUStringView),
    ]


class WGPUComputeState(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("module", WGPUShaderModule),
        ("entryPoint", WGPUStringView),
        ("constantCount", SIZE_T),
        ("constants", ctypes.c_void_p),
    ]


class WGPUComputePipelineDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("layout", WGPUPipelineLayout),
        ("compute", WGPUComputeState),
    ]


class WGPUBufferDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("usage", WGPUBufferUsageFlags),
        ("size", ctypes.c_uint64),
        ("mappedAtCreation", WGPUBool),
    ]


class WGPUBindGroupEntry(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("binding", ctypes.c_uint32),
        ("buffer", WGPUBuffer),
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("sampler", WGPUSampler),
        ("textureView", WGPUTextureView),
    ]


class WGPUBindGroupDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("layout", WGPUBindGroupLayout),
        ("entryCount", SIZE_T),
        ("entries", ctypes.POINTER(WGPUBindGroupEntry)),
    ]


class WGPUBufferBindingLayout(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("type", ctypes.c_uint32),
        ("hasDynamicOffset", WGPUBool),
        ("minBindingSize", ctypes.c_uint64),
    ]


class WGPUSamplerBindingLayout(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("type", ctypes.c_uint32),
    ]


class WGPUTextureBindingLayout(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("sampleType", ctypes.c_uint32),
        ("viewDimension", ctypes.c_uint32),
        ("multisampled", WGPUBool),
    ]


class WGPUStorageTextureBindingLayout(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("access", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
        ("viewDimension", ctypes.c_uint32),
    ]


class WGPUBindGroupLayoutEntry(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("binding", ctypes.c_uint32),
        ("visibility", WGPUShaderStageFlags),
        ("bindingArraySize", ctypes.c_uint32),
        ("buffer", WGPUBufferBindingLayout),
        ("sampler", WGPUSamplerBindingLayout),
        ("texture", WGPUTextureBindingLayout),
        ("storageTexture", WGPUStorageTextureBindingLayout),
    ]


class WGPUBindGroupLayoutDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("entryCount", SIZE_T),
        ("entries", ctypes.POINTER(WGPUBindGroupLayoutEntry)),
    ]


class WGPUPipelineLayoutDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("bindGroupLayoutCount", SIZE_T),
        ("bindGroupLayouts", ctypes.POINTER(WGPUBindGroupLayout)),
        ("immediateSize", ctypes.c_uint32),
    ]


class WGPUCommandEncoderDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
    ]


class WGPUCommandBufferDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
    ]


class WGPUComputePassDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("timestampWrites", ctypes.c_void_p),
    ]


class WGPUPassTimestampWrites(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.c_void_p),
        ("querySet", ctypes.c_void_p),  # WGPUQuerySet
        ("beginningOfPassWriteIndex", ctypes.c_uint32),
        ("endOfPassWriteIndex", ctypes.c_uint32),
    ]


class WGPUQuerySetDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.c_void_p),
        ("label", WGPUStringView),
        ("type", ctypes.c_uint32),  # WGPUQueryType
        ("count", ctypes.c_uint32),
    ]


# ============================================================================
# Function prototype setup
# ============================================================================

def _setup_prototypes(lib):
    """Set argument and return types for all used Dawn functions."""

    # wgpuCreateInstance
    lib.wgpuCreateInstance.argtypes = [ctypes.POINTER(WGPUInstanceDescriptor)]
    lib.wgpuCreateInstance.restype = WGPUInstance

    # wgpuInstanceRequestAdapter
    lib.wgpuInstanceRequestAdapter.argtypes = [
        WGPUInstance,
        ctypes.POINTER(WGPURequestAdapterOptions),
        WGPURequestAdapterCallbackInfo,
    ]
    lib.wgpuInstanceRequestAdapter.restype = WGPUFuture

    # wgpuInstanceWaitAny
    lib.wgpuInstanceWaitAny.argtypes = [
        WGPUInstance, SIZE_T,
        ctypes.POINTER(WGPUFutureWaitInfo),
        ctypes.c_uint64,
    ]
    lib.wgpuInstanceWaitAny.restype = ctypes.c_uint32  # WGPUWaitStatus

    # wgpuInstanceProcessEvents
    lib.wgpuInstanceProcessEvents.argtypes = [WGPUInstance]
    lib.wgpuInstanceProcessEvents.restype = None

    # wgpuAdapterCreateDevice (Dawn synchronous extension)
    lib.wgpuAdapterCreateDevice.argtypes = [
        WGPUAdapter, ctypes.POINTER(WGPUDeviceDescriptor)
    ]
    lib.wgpuAdapterCreateDevice.restype = WGPUDevice

    # wgpuAdapterGetLimits
    lib.wgpuAdapterGetLimits.argtypes = [
        WGPUAdapter, ctypes.POINTER(WGPULimits)
    ]
    lib.wgpuAdapterGetLimits.restype = ctypes.c_uint32  # WGPUStatus

    # wgpuAdapterRelease / wgpuAdapterAddRef
    lib.wgpuAdapterRelease.argtypes = [WGPUAdapter]
    lib.wgpuAdapterRelease.restype = None

    # wgpuAdapterHasFeature
    lib.wgpuAdapterHasFeature.argtypes = [WGPUAdapter, ctypes.c_uint32]
    lib.wgpuAdapterHasFeature.restype = ctypes.c_uint32  # WGPUBool

    # wgpuAdapterGetInfo
    lib.wgpuAdapterGetInfo.argtypes = [WGPUAdapter, ctypes.POINTER(WGPUAdapterInfo)]
    lib.wgpuAdapterGetInfo.restype = ctypes.c_uint32  # WGPUStatus

    # wgpuDeviceGetQueue
    lib.wgpuDeviceGetQueue.argtypes = [WGPUDevice]
    lib.wgpuDeviceGetQueue.restype = WGPUQueue

    # wgpuDeviceCreateShaderModule
    lib.wgpuDeviceCreateShaderModule.argtypes = [
        WGPUDevice, ctypes.POINTER(WGPUShaderModuleDescriptor)
    ]
    lib.wgpuDeviceCreateShaderModule.restype = WGPUShaderModule

    # wgpuDeviceCreateComputePipeline
    lib.wgpuDeviceCreateComputePipeline.argtypes = [
        WGPUDevice, ctypes.POINTER(WGPUComputePipelineDescriptor)
    ]
    lib.wgpuDeviceCreateComputePipeline.restype = WGPUComputePipeline

    # wgpuDeviceCreateComputePipelineAsync (if exported by Dawn)
    if hasattr(lib, 'wgpuDeviceCreateComputePipelineAsync'):
        lib.wgpuDeviceCreateComputePipelineAsync.argtypes = [
            WGPUDevice,
            ctypes.POINTER(WGPUComputePipelineDescriptor),
            WGPUCreateComputePipelineAsyncCallbackInfo,
        ]
        lib.wgpuDeviceCreateComputePipelineAsync.restype = WGPUFuture

    # wgpuDeviceCreateBuffer
    lib.wgpuDeviceCreateBuffer.argtypes = [
        WGPUDevice, ctypes.POINTER(WGPUBufferDescriptor)
    ]
    lib.wgpuDeviceCreateBuffer.restype = WGPUBuffer

    # wgpuDeviceCreateBindGroupLayout
    lib.wgpuDeviceCreateBindGroupLayout.argtypes = [
        WGPUDevice, ctypes.POINTER(WGPUBindGroupLayoutDescriptor)
    ]
    lib.wgpuDeviceCreateBindGroupLayout.restype = WGPUBindGroupLayout

    # wgpuDeviceCreatePipelineLayout
    lib.wgpuDeviceCreatePipelineLayout.argtypes = [
        WGPUDevice, ctypes.POINTER(WGPUPipelineLayoutDescriptor)
    ]
    lib.wgpuDeviceCreatePipelineLayout.restype = WGPUPipelineLayout

    # wgpuDeviceCreateBindGroup
    lib.wgpuDeviceCreateBindGroup.argtypes = [
        WGPUDevice, ctypes.POINTER(WGPUBindGroupDescriptor)
    ]
    lib.wgpuDeviceCreateBindGroup.restype = WGPUBindGroup

    # wgpuDeviceCreateCommandEncoder
    lib.wgpuDeviceCreateCommandEncoder.argtypes = [
        WGPUDevice, ctypes.POINTER(WGPUCommandEncoderDescriptor)
    ]
    lib.wgpuDeviceCreateCommandEncoder.restype = WGPUCommandEncoder

    # wgpuCommandEncoderBeginComputePass
    lib.wgpuCommandEncoderBeginComputePass.argtypes = [
        WGPUCommandEncoder, ctypes.POINTER(WGPUComputePassDescriptor)
    ]
    lib.wgpuCommandEncoderBeginComputePass.restype = WGPUComputePassEncoder

    # wgpuComputePassEncoderSetPipeline
    lib.wgpuComputePassEncoderSetPipeline.argtypes = [
        WGPUComputePassEncoder, WGPUComputePipeline
    ]
    lib.wgpuComputePassEncoderSetPipeline.restype = None

    # wgpuComputePassEncoderSetBindGroup
    lib.wgpuComputePassEncoderSetBindGroup.argtypes = [
        WGPUComputePassEncoder, ctypes.c_uint32,
        WGPUBindGroup, SIZE_T, ctypes.POINTER(ctypes.c_uint32)
    ]
    lib.wgpuComputePassEncoderSetBindGroup.restype = None

    # wgpuComputePassEncoderDispatchWorkgroups
    lib.wgpuComputePassEncoderDispatchWorkgroups.argtypes = [
        WGPUComputePassEncoder, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
    ]
    lib.wgpuComputePassEncoderDispatchWorkgroups.restype = None

    # wgpuComputePassEncoderEnd
    lib.wgpuComputePassEncoderEnd.argtypes = [WGPUComputePassEncoder]
    lib.wgpuComputePassEncoderEnd.restype = None

    # wgpuCommandEncoderCopyBufferToBuffer
    lib.wgpuCommandEncoderCopyBufferToBuffer.argtypes = [
        WGPUCommandEncoder, WGPUBuffer, ctypes.c_uint64,
        WGPUBuffer, ctypes.c_uint64, ctypes.c_uint64
    ]
    lib.wgpuCommandEncoderCopyBufferToBuffer.restype = None

    # wgpuCommandEncoderFinish
    lib.wgpuCommandEncoderFinish.argtypes = [
        WGPUCommandEncoder, ctypes.POINTER(WGPUCommandBufferDescriptor)
    ]
    lib.wgpuCommandEncoderFinish.restype = WGPUCommandBuffer

    # wgpuQueueSubmit
    lib.wgpuQueueSubmit.argtypes = [
        WGPUQueue, SIZE_T, ctypes.POINTER(WGPUCommandBuffer)
    ]
    lib.wgpuQueueSubmit.restype = None

    # wgpuQueueWriteBuffer
    lib.wgpuQueueWriteBuffer.argtypes = [
        WGPUQueue, WGPUBuffer, ctypes.c_uint64,
        ctypes.c_void_p, SIZE_T
    ]
    lib.wgpuQueueWriteBuffer.restype = None

    # wgpuBufferMapAsync
    lib.wgpuBufferMapAsync.argtypes = [
        WGPUBuffer, WGPUMapModeFlags, SIZE_T, SIZE_T,
        WGPUBufferMapCallbackInfo,
    ]
    lib.wgpuBufferMapAsync.restype = WGPUFuture

    # wgpuBufferGetConstMappedRange
    lib.wgpuBufferGetConstMappedRange.argtypes = [
        WGPUBuffer, SIZE_T, SIZE_T
    ]
    lib.wgpuBufferGetConstMappedRange.restype = ctypes.c_void_p

    # wgpuBufferUnmap
    lib.wgpuBufferUnmap.argtypes = [WGPUBuffer]
    lib.wgpuBufferUnmap.restype = None

    # wgpuBufferDestroy
    lib.wgpuBufferDestroy.argtypes = [WGPUBuffer]
    lib.wgpuBufferDestroy.restype = None

    # wgpuBufferRelease
    lib.wgpuBufferRelease.argtypes = [WGPUBuffer]
    lib.wgpuBufferRelease.restype = None

    # wgpuShaderModuleRelease
    lib.wgpuShaderModuleRelease.argtypes = [WGPUShaderModule]
    lib.wgpuShaderModuleRelease.restype = None

    # wgpuComputePipelineRelease
    lib.wgpuComputePipelineRelease.argtypes = [WGPUComputePipeline]
    lib.wgpuComputePipelineRelease.restype = None

    # wgpuBindGroupRelease
    lib.wgpuBindGroupRelease.argtypes = [WGPUBindGroup]
    lib.wgpuBindGroupRelease.restype = None

    # wgpuBindGroupLayoutRelease
    lib.wgpuBindGroupLayoutRelease.argtypes = [WGPUBindGroupLayout]
    lib.wgpuBindGroupLayoutRelease.restype = None

    # wgpuPipelineLayoutRelease
    lib.wgpuPipelineLayoutRelease.argtypes = [WGPUPipelineLayout]
    lib.wgpuPipelineLayoutRelease.restype = None

    # wgpuCommandBufferRelease
    lib.wgpuCommandBufferRelease.argtypes = [WGPUCommandBuffer]
    lib.wgpuCommandBufferRelease.restype = None

    # wgpuComputePassEncoderRelease
    lib.wgpuComputePassEncoderRelease.argtypes = [WGPUComputePassEncoder]
    lib.wgpuComputePassEncoderRelease.restype = None

    # wgpuCommandEncoderRelease
    lib.wgpuCommandEncoderRelease.argtypes = [WGPUCommandEncoder]
    lib.wgpuCommandEncoderRelease.restype = None

    # wgpuDeviceRelease
    lib.wgpuDeviceRelease.argtypes = [WGPUDevice]
    lib.wgpuDeviceRelease.restype = None

    # wgpuQueueRelease
    lib.wgpuQueueRelease.argtypes = [WGPUQueue]
    lib.wgpuQueueRelease.restype = None

    # wgpuInstanceRelease
    lib.wgpuInstanceRelease.argtypes = [WGPUInstance]
    lib.wgpuInstanceRelease.restype = None

    # wgpuDeviceTick
    lib.wgpuDeviceTick.argtypes = [WGPUDevice]
    lib.wgpuDeviceTick.restype = WGPUBool

    # --- Timestamp query APIs ---
    WGPUQuerySet = ctypes.c_void_p

    # wgpuDeviceCreateQuerySet
    lib.wgpuDeviceCreateQuerySet.argtypes = [WGPUDevice, ctypes.c_void_p]
    lib.wgpuDeviceCreateQuerySet.restype = WGPUQuerySet

    # wgpuCommandEncoderResolveQuerySet
    lib.wgpuCommandEncoderResolveQuerySet.argtypes = [
        WGPUCommandEncoder, WGPUQuerySet,
        ctypes.c_uint32, ctypes.c_uint32,  # firstQuery, queryCount
        WGPUBuffer, ctypes.c_uint64,  # destination, destinationOffset
    ]
    lib.wgpuCommandEncoderResolveQuerySet.restype = None

    # wgpuQuerySetDestroy
    lib.wgpuQuerySetDestroy.argtypes = [WGPUQuerySet]
    lib.wgpuQuerySetDestroy.restype = None

    # wgpuQuerySetRelease
    lib.wgpuQuerySetRelease.argtypes = [WGPUQuerySet]
    lib.wgpuQuerySetRelease.restype = None


# ============================================================================
# Numpy / WGSL type mappings
# ============================================================================

WGSL_TYPE_TO_NUMPY = {
    'f32': np.float32,
    'f16': np.float16,
    'i32': np.int32,
    'u32': np.uint32,
}

WGSL_TYPE_TO_STRUCT_FMT = {
    'f32': '<f',
    'f16': '<e',
    'i32': '<i',
    'u32': '<I',
}


# ============================================================================
# GPUBuffer — handle to GPU-resident data
# ============================================================================

class GPUBuffer:
    """Handle to a GPU-resident buffer for zero-copy kernel inputs/outputs.

    When passed as a value in the ``buffers`` dict of ``run_kernel()``, the
    runner skips the CPU→GPU upload and uses the GPU buffer directly.  When
    returned by ``run_kernel()`` via ``gpu_outputs``, the result stays on
    GPU and can be fed into subsequent kernel calls without readback.
    """
    __slots__ = ('_runner', 'handle', 'size', 'dtype', 'shape', '_owned')

    def __init__(self, runner, handle, size, dtype=np.float32,
                 shape=None, owned=True):
        self._runner = runner
        self.handle = handle       # WGPUBuffer
        self.size = size           # bytes
        self.dtype = dtype
        self.shape = shape
        self._owned = owned        # if True, __del__ destroys & releases

    @property
    def nbytes(self):
        return self.size

    def __del__(self):
        if self._owned and self.handle:
            runner = self._runner
            if runner and runner._lib:
                try:
                    runner._lib.wgpuBufferDestroy(self.handle)
                    runner._lib.wgpuBufferRelease(self.handle)
                except Exception:
                    pass
            self.handle = None


# ============================================================================
# DawnRunner — main runtime class
# ============================================================================

class DawnRunner:
    """Execute WGSL compute shaders on the GPU via Dawn's native WebGPU API."""

    def __init__(self):
        self._lib = _load_dawn()
        if self._lib is None:
            raise RuntimeError(
                "Dawn WebGPU library not found. "
                "Build Dawn or set DAWN_PATH environment variable. "
                "Expected: webgpu_dawn.dll / libwebgpu_dawn.so"
            )

        # Create instance with TimedWaitAny feature enabled
        features = (ctypes.c_uint32 * 1)(0x00000001)  # WGPUInstanceFeatureName_TimedWaitAny
        desc = WGPUInstanceDescriptor()
        desc.nextInChain = None
        desc.requiredFeatureCount = 1
        desc.requiredFeatures = ctypes.cast(features, ctypes.c_void_p)
        desc.requiredLimits = None
        self._instance = self._lib.wgpuCreateInstance(ctypes.byref(desc))
        if not self._instance:
            raise RuntimeError("Failed to create Dawn WebGPU instance")

        # Request adapter (async with WaitAny)
        self._adapter = None
        self._adapter_error = None

        @RequestAdapterCallback
        def on_adapter(status, adapter, message, ud1, ud2):
            if status == WGPURequestAdapterStatus.Success:
                self._adapter = adapter
            else:
                msg = ""
                if message.data:
                    msg = message.data.decode("utf-8", errors="replace")
                self._adapter_error = msg

        # Keep callback alive
        self._adapter_cb = on_adapter

        # -- Adapter toggles (matching onnxruntime WebGPU EP) --
        # use_dxc: Shader Model 6+ (required for native f16 on D3D12)
        # allow_unsafe_apis: enable Chrome experimental features
        adapter_toggle_names = [b"use_dxc", b"allow_unsafe_apis"]
        self._adapter_toggle_ptrs = (ctypes.c_char_p * len(adapter_toggle_names))(
            *adapter_toggle_names)
        adapter_toggles = WGPUDawnTogglesDescriptor()
        adapter_toggles.chain.next = None
        adapter_toggles.chain.sType = WGPUSType.DawnTogglesDescriptor
        adapter_toggles.enabledToggleCount = len(adapter_toggle_names)
        adapter_toggles.enabledToggles = self._adapter_toggle_ptrs
        adapter_toggles.disabledToggleCount = 0
        adapter_toggles.disabledToggles = None
        # prevent GC
        self._adapter_toggles = adapter_toggles

        opts = WGPURequestAdapterOptions()
        opts.nextInChain = ctypes.cast(ctypes.pointer(adapter_toggles), ctypes.POINTER(WGPUChainedStruct))
        opts.featureLevel = WGPUFeatureLevel.Core

        # GPU selection via DAWN_GPU environment variable:
        #   DAWN_GPU=0 or "high" → HighPerformance (discrete GPU, default)
        #   DAWN_GPU=1 or "low"  → LowPower (integrated GPU)
        gpu_env = os.environ.get("DAWN_GPU", "").strip().lower()
        if gpu_env in ("1", "low", "integrated"):
            opts.powerPreference = WGPUPowerPreference.LowPower
        else:
            opts.powerPreference = WGPUPowerPreference.HighPerformance

        opts.forceFallbackAdapter = 0
        opts.backendType = WGPUBackendType.D3D12  # Use D3D12 on Windows
        opts.compatibleSurface = None

        cb_info = WGPURequestAdapterCallbackInfo()
        cb_info.nextInChain = None
        cb_info.mode = WGPUCallbackMode.WaitAnyOnly
        cb_info.callback = on_adapter
        cb_info.userdata1 = None
        cb_info.userdata2 = None

        future = self._lib.wgpuInstanceRequestAdapter(
            self._instance, ctypes.byref(opts), cb_info
        )

        # Wait for adapter
        wait_info = WGPUFutureWaitInfo()
        wait_info.future = future
        wait_info.completed = 0

        status = self._lib.wgpuInstanceWaitAny(
            self._instance, 1, ctypes.byref(wait_info), ctypes.c_uint64(-1)  # UINT64_MAX = infinite timeout
        )

        if not self._adapter:
            err = self._adapter_error or "unknown error"
            raise RuntimeError(f"Failed to obtain WebGPU adapter from Dawn: {err}")

        # Query adapter limits
        adapter_limits = WGPULimits()
        ctypes.memset(ctypes.byref(adapter_limits), 0, ctypes.sizeof(WGPULimits))
        adapter_limits.nextInChain = None
        self._lib.wgpuAdapterGetLimits(self._adapter, ctypes.byref(adapter_limits))
        self._adapter_limits = adapter_limits

        # Create device (synchronous Dawn extension)
        # Detect ShaderF16 and Subgroups features.
        # Dawn's enum values shifted when CoreFeaturesAndLimits (0x01) was added,
        # so we probe both old and new IDs for compatibility with different
        # Dawn builds.
        WGPUFeatureName = ctypes.c_uint32
        # New API: ShaderF16=0x0B, Subgroups=0x12
        # Old API: ShaderF16=0x0A, Subgroups=0x11
        FEATURE_SHADER_F16_IDS = [0x0000000B, 0x0000000A]
        FEATURE_SUBGROUPS_IDS = [0x00000012, 0x00000011]
        requested_features = []
        self._shader_f16_id = None
        self._subgroups_id = None
        self._timestamp_query_id = None
        for fid in FEATURE_SHADER_F16_IDS:
            if self._lib.wgpuAdapterHasFeature(self._adapter, fid):
                requested_features.append(fid)
                self._shader_f16_id = fid
                break
        for fid in FEATURE_SUBGROUPS_IDS:
            if self._lib.wgpuAdapterHasFeature(self._adapter, fid):
                requested_features.append(fid)
                self._subgroups_id = fid
                break
        # TimestampQuery: new=0x09, old=0x03
        FEATURE_TIMESTAMP_IDS = [0x00000009, 0x00000003]
        for fid in FEATURE_TIMESTAMP_IDS:
            if self._lib.wgpuAdapterHasFeature(self._adapter, fid):
                requested_features.append(fid)
                self._timestamp_query_id = fid
                break

        # Request adapter's actual limits instead of conservative defaults.
        # Copy the entire adapter limits struct — this handles both "max"
        # fields (higher = better) and "min" alignment fields (must not
        # request a value *lower* than supported).
        required_limits = WGPULimits()
        ctypes.memmove(ctypes.byref(required_limits),
                       ctypes.byref(adapter_limits),
                       ctypes.sizeof(WGPULimits))
        required_limits.nextInChain = None
        # Keep reference alive
        self._required_limits = required_limits

        # -- Device toggles (matching onnxruntime WebGPU EP) --
        dev_enabled_names = [b"skip_validation", b"disable_robustness",
                             b"d3d_disable_ieee_strictness"]
        dev_disabled_names = [b"lazy_clear_resource_on_first_use",
                              b"timestamp_quantization"]
        self._dev_enabled_ptrs = (ctypes.c_char_p * len(dev_enabled_names))(
            *dev_enabled_names)
        self._dev_disabled_ptrs = (ctypes.c_char_p * len(dev_disabled_names))(
            *dev_disabled_names)
        dev_toggles = WGPUDawnTogglesDescriptor()
        dev_toggles.chain.next = None
        dev_toggles.chain.sType = WGPUSType.DawnTogglesDescriptor
        dev_toggles.enabledToggleCount = len(dev_enabled_names)
        dev_toggles.enabledToggles = self._dev_enabled_ptrs
        dev_toggles.disabledToggleCount = len(dev_disabled_names)
        dev_toggles.disabledToggles = self._dev_disabled_ptrs
        self._dev_toggles = dev_toggles

        dev_desc = WGPUDeviceDescriptor()
        ctypes.memset(ctypes.byref(dev_desc), 0, ctypes.sizeof(WGPUDeviceDescriptor))
        dev_desc.nextInChain = ctypes.cast(ctypes.pointer(dev_toggles), ctypes.POINTER(WGPUChainedStruct))
        dev_desc.label = WGPUStringView.from_str("triton")

        # Set uncaptured error callback to catch validation errors
        @UncapturedErrorCallback
        def on_uncaptured_error(device, error_type, message, ud1, ud2):
            try:
                msg_str = message.data[:message.length].decode('utf-8', errors='replace') if message.data and message.length > 0 else "<no message>"
            except Exception:
                msg_str = "<failed to decode>"
            error_names = {0: "NoError", 1: "Validation", 2: "OutOfMemory", 3: "Internal", 4: "Unknown", 5: "DeviceLost"}
            print(f"[DAWN ERROR] type={error_names.get(error_type, error_type)} msg={msg_str[:500]}", flush=True)
        self._uncaptured_error_cb = on_uncaptured_error
        dev_desc.uncapturedErrorCallbackInfo.callback = ctypes.cast(
            on_uncaptured_error, ctypes.c_void_p)
        dev_desc.uncapturedErrorCallbackInfo.userdata1 = None
        dev_desc.uncapturedErrorCallbackInfo.userdata2 = None
        if requested_features:
            features_array = (WGPUFeatureName * len(requested_features))(
                *requested_features)
            dev_desc.requiredFeatureCount = len(requested_features)
            dev_desc.requiredFeatures = ctypes.cast(features_array, ctypes.c_void_p)
        else:
            dev_desc.requiredFeatureCount = 0
            dev_desc.requiredFeatures = None
        dev_desc.requiredLimits = ctypes.pointer(required_limits)
        self._has_subgroups = self._subgroups_id is not None
        self._has_f16 = self._shader_f16_id is not None
        self._has_timestamp_query = self._timestamp_query_id is not None

        self._device = self._lib.wgpuAdapterCreateDevice(
            self._adapter, ctypes.byref(dev_desc)
        )
        if not self._device:
            raise RuntimeError("Failed to create Dawn WebGPU device")

        self._queue = self._lib.wgpuDeviceGetQueue(self._device)
        self._has_async_compute_pipeline = hasattr(
            self._lib, 'wgpuDeviceCreateComputePipelineAsync')

        # Cache for pipelines and shader modules
        self._pipeline_cache = {}  # wgsl_hash -> (shader_module, pipeline, bg_layout, pipeline_layout)
        self._pipeline_cache_lock = threading.RLock()
        self._pipeline_executor = ThreadPoolExecutor(
            max_workers=max(1, min(8, (os.cpu_count() or 4))))
        # Cache for GPU buffers to avoid per-call alloc/dealloc
        self._buffer_cache = {}  # (name, size, usage) -> WGPUBuffer
        # Toggle pool for gpu_outputs — avoids fresh allocation each call
        self._gpu_out_toggles = {}  # pool_key -> 0 or 1
        # Track total GPU memory allocated via this runner
        self._total_gpu_bytes = 0
        self._gpu_alloc_count = 0

        # Size-class buffer pool: reuses freed buffers by rounded size
        # instead of requiring exact (name, size) match.
        self._pool_free = {}      # rounded_size -> [WGPUBuffer, ...]
        self._pool_alloc = 0      # total buffers created via pool
        self._pool_reuse = 0      # total buffers reused from pool
        self._pool_bytes = 0      # total bytes allocated via pool

    def gpu_memory_stats(self) -> dict:
        """Return GPU memory usage statistics.

        Tracks all buffers allocated via this runner:
        - buffer_cache: internal kernel I/O buffers (reused per dispatch)
        - upload_to_gpu: weight and data buffers (permanent, owned)
        - pool: size-class pooled transient buffers
        """
        cache_bytes = sum(size for (_, size, _) in self._buffer_cache.keys())
        pool_free_count = sum(len(v) for v in self._pool_free.values())
        return {
            'total_allocated_mb': self._total_gpu_bytes / 1024 / 1024,
            'buffer_cache_entries': len(self._buffer_cache),
            'buffer_cache_mb': cache_bytes / 1024 / 1024,
            'pipeline_cache_entries': len(self._pipeline_cache),
            'alloc_count': self._gpu_alloc_count,
            'pool_alloc': self._pool_alloc,
            'pool_reuse': self._pool_reuse,
            'pool_free': pool_free_count,
            'pool_bytes_mb': self._pool_bytes / 1024 / 1024,
        }

    @property
    def adapter_info(self) -> dict:
        """Query adapter info: vendor, device name, backend type."""
        if hasattr(self, '_adapter_info_cache'):
            return self._adapter_info_cache
        info = WGPUAdapterInfo()
        ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(WGPUAdapterInfo))
        self._lib.wgpuAdapterGetInfo(self._adapter, ctypes.byref(info))

        def _sv(sv):
            if sv.data and sv.length > 0:
                return sv.data[:sv.length].decode('utf-8', errors='replace')
            return ''

        backend_names = {
            WGPUBackendType.D3D11: 'D3D11', WGPUBackendType.D3D12: 'D3D12',
            WGPUBackendType.Metal: 'Metal', WGPUBackendType.Vulkan: 'Vulkan',
            WGPUBackendType.WebGPU: 'WebGPU', WGPUBackendType.Null: 'Null',
        }
        self._adapter_info_cache = {
            'vendor': _sv(info.vendor),
            'architecture': _sv(info.architecture),
            'device': _sv(info.device),
            'description': _sv(info.description),
            'backend': backend_names.get(info.backendType, f'Unknown({info.backendType})'),
            'vendorID': info.vendorID,
            'deviceID': info.deviceID,
        }
        return self._adapter_info_cache

    @property
    def adapter_info_str(self) -> str:
        """Human-readable GPU adapter string."""
        info = self.adapter_info
        return f"{info['description']} ({info['backend']})"

    # -- Batched dispatch ------------------------------------------------------

    def begin_batch(self):
        """Start a batched dispatch session.

        While batching, run_kernel() with gpu_outputs accumulates
        compute passes in a shared command encoder instead of submitting
        individually. Call end_batch() to submit all at once.

        This eliminates per-dispatch wgpuQueueSubmit overhead (~0.1ms each)
        and can significantly speed up sequences of GPU operations.
        """
        lib = self._lib
        enc_desc = WGPUCommandEncoderDescriptor()
        enc_desc.nextInChain = None
        enc_desc.label = WGPUStringView.from_str("batch")
        self._batch_encoder = lib.wgpuDeviceCreateCommandEncoder(
            self._device, ctypes.byref(enc_desc))
        self._batch_readbacks = []  # list of (gpu_buf, size, bindings) to readback

    def end_batch(self, readback_buffers=None):
        """Submit all batched dispatches and optionally readback results."""
        if not hasattr(self, '_batch_encoder') or self._batch_encoder is None:
            return {}

        lib = self._lib
        encoder = self._batch_encoder

        # Add readback copies for requested buffers
        results = {}
        readback_mapping = {}
        if readback_buffers:
            readback_usage = BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST
            for i, gpu_buf in enumerate(readback_buffers):
                size = gpu_buf.size
                rb_name = f"__batch_rb_{i}__"
                rb_buf = self._get_or_create_buffer(
                    rb_name, size, readback_usage)
                lib.wgpuCommandEncoderCopyBufferToBuffer(
                    encoder, gpu_buf.handle, 0, rb_buf, 0, size)
                readback_mapping[rb_name] = (rb_buf, size, gpu_buf)

        # Finish and submit
        cb_desc = WGPUCommandBufferDescriptor()
        cb_desc.nextInChain = None
        cb_desc.label = WGPUStringView.from_str("")
        cmd_buf = lib.wgpuCommandEncoderFinish(encoder, ctypes.byref(cb_desc))
        cmd_bufs = (ctypes.c_void_p * 1)(cmd_buf)
        lib.wgpuQueueSubmit(
            self._queue, 1,
            ctypes.cast(cmd_bufs, ctypes.POINTER(WGPUCommandBuffer)))

        # Release encoder and command buffer
        lib.wgpuCommandEncoderRelease(encoder)
        lib.wgpuCommandBufferRelease(cmd_buf)

        # Release accumulated batch compute passes and bind groups
        if hasattr(self, '_batch_cleanup'):
            for compute_pass, bind_group in self._batch_cleanup:
                lib.wgpuComputePassEncoderRelease(compute_pass)
                lib.wgpuBindGroupRelease(bind_group)
            self._batch_cleanup = []

        # Read back requested buffers
        for rb_name, (rb_buf, size, gpu_buf) in readback_mapping.items():
            data = self._map_and_read(rb_buf, size, gpu_buf.dtype)
            results[id(gpu_buf)] = data

        self._batch_encoder = None
        self._batch_readbacks = []
        return results

    def _map_and_read(self, rb_buf, size, dtype):
        """Map a readback buffer and return numpy array."""
        lib = self._lib
        map_done = [False]
        map_status = [0]

        @BufferMapCallback
        def on_map(status, message, ud1, ud2, _md=map_done, _ms=map_status):
            _md[0] = True
            _ms[0] = status

        self._map_cb = on_map
        cb_info = WGPUBufferMapCallbackInfo()
        cb_info.nextInChain = None
        cb_info.mode = WGPUCallbackMode.WaitAnyOnly
        cb_info.callback = on_map
        cb_info.userdata1 = None
        cb_info.userdata2 = None

        future = lib.wgpuBufferMapAsync(
            rb_buf, MAP_MODE_READ, 0, size, cb_info)
        wait_info = WGPUFutureWaitInfo()
        wait_info.future = future
        wait_info.completed = 0
        lib.wgpuInstanceWaitAny(
            self._instance, 1, ctypes.byref(wait_info),
            ctypes.c_uint64(-1))

        if map_status[0] != WGPUMapAsyncStatus.Success:
            raise RuntimeError(f"Batch readback map failed: {map_status[0]}")

        data_ptr = lib.wgpuBufferGetConstMappedRange(rb_buf, 0, size)
        np_dtype = dtype if dtype else np.float32
        result = np.ctypeslib.as_array(
            (ctypes.c_uint8 * size).from_address(data_ptr),
            shape=(size,)
        ).view(np_dtype).copy()

        lib.wgpuBufferUnmap(rb_buf)
        return result

    @property
    def is_batching(self):
        """Whether we're currently in a batch dispatch session."""
        return hasattr(self, '_batch_encoder') and self._batch_encoder is not None

    @property
    def max_storage_buffer_binding_size(self) -> int:
        """Adapter's maxStorageBufferBindingSize in bytes."""
        return int(self._adapter_limits.maxStorageBufferBindingSize)

    @property
    def max_compute_invocations_per_workgroup(self) -> int:
        """Adapter's maxComputeInvocationsPerWorkgroup."""
        return int(self._adapter_limits.maxComputeInvocationsPerWorkgroup)

    @property
    def has_subgroups(self) -> bool:
        """Whether the adapter supports native WebGPU Subgroups."""
        return self._has_subgroups

    @property
    def has_f16(self) -> bool:
        """Whether the adapter supports ShaderF16."""
        return self._has_f16

    @property
    def has_timestamp_query(self) -> bool:
        """Whether the adapter supports TimestampQuery."""
        return self._has_timestamp_query

    # -- Fast dispatch API ----------------------------------------------------
    #
    # Pre-create bind groups and submit multiple dispatches with minimal
    # per-dispatch Python/ctypes overhead.  Used by the fast decode path
    # to avoid repeated buffer allocation, bind group creation, and
    # compute pass begin/end calls.

    def get_pipeline_info(self, wgsl_code, buffer_bindings, param_fields):
        """Get cached pipeline and bind group layout for compiled WGSL.

        Returns (pipeline, bg_layout) tuple.
        """
        _, pipeline, bg_layout, _ = self._get_or_create_pipeline(
            wgsl_code, buffer_bindings, param_fields)
        return pipeline, bg_layout

    def create_compute_pipeline_async(self, wgsl_code, buffer_bindings,
                                      param_fields):
        """Schedule asynchronous pipeline creation on a worker thread.

        Returns a Future that resolves to
        (shader_module, pipeline, bg_layout, pipeline_layout).
        """
        return self._pipeline_executor.submit(
            self._get_or_create_pipeline,
            wgsl_code, buffer_bindings, param_fields)

    def prefetch_pipelines_async(self, pipeline_specs, max_workers=None):
        """Compile multiple pipelines concurrently.

        Args:
            pipeline_specs: iterable of (wgsl_code, buffer_bindings, param_fields)
            max_workers: optional worker cap for this prefetch call

        Returns:
            Number of pipeline specs submitted for compilation.
        """
        if not pipeline_specs:
            return 0

        # Deduplicate by WGSL hash key.
        dedup = {}
        for wgsl_code, buffer_bindings, param_fields in pipeline_specs:
            key = self._pipeline_cache_key(wgsl_code)
            dedup[key] = (wgsl_code, buffer_bindings, param_fields)

        pending_specs = []
        with self._pipeline_cache_lock:
            for key, spec in dedup.items():
                if key not in self._pipeline_cache:
                    pending_specs.append(spec)

        if not pending_specs:
            return 0

        if max_workers is not None:
            workers = max(1, int(max_workers))
            executor = ThreadPoolExecutor(max_workers=workers)
            owns_executor = True
        else:
            executor = self._pipeline_executor
            owns_executor = False

        futures = [
            executor.submit(self._get_or_create_pipeline, wgsl, bbs, pfs)
            for wgsl, bbs, pfs in pending_specs
        ]
        try:
            for fut in as_completed(futures):
                fut.result()
        finally:
            if owns_executor:
                executor.shutdown(wait=True)

        return len(pending_specs)

    def create_gpu_buffer(self, name, size_bytes):
        """Create a named GPU storage buffer.

        Returns the raw WGPUBuffer handle (ctypes.c_void_p).
        Uses the existing buffer cache — safe to call multiple times
        with the same name/size.
        """
        usage = (BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_SRC
                 | BUFFER_USAGE_COPY_DST)
        return self._get_or_create_buffer(name, size_bytes, usage)

    def create_bind_group(self, bg_layout, entries):
        """Create a bind group from raw buffer handles.

        Args:
            bg_layout: WGPUBindGroupLayout handle
            entries: list of (binding_index, buffer_handle, size_bytes) tuples

        Returns:
            WGPUBindGroup handle (caller must release when done)
        """
        lib = self._lib
        n = len(entries)
        BindEntryArray = WGPUBindGroupEntry * n
        bind_entries = BindEntryArray()
        for i, (binding, buf_handle, size) in enumerate(entries):
            ctypes.memset(ctypes.byref(bind_entries[i]), 0,
                         ctypes.sizeof(WGPUBindGroupEntry))
            bind_entries[i].binding = binding
            bind_entries[i].buffer = buf_handle
            bind_entries[i].offset = 0
            bind_entries[i].size = size

        bg_desc = WGPUBindGroupDescriptor()
        bg_desc.nextInChain = None
        bg_desc.label = WGPUStringView.from_str("")
        bg_desc.layout = bg_layout
        bg_desc.entryCount = n
        bg_desc.entries = bind_entries

        return lib.wgpuDeviceCreateBindGroup(
            self._device, ctypes.byref(bg_desc))

    def write_buffer(self, buf_handle, data_bytes):
        """Write raw bytes to a GPU buffer."""
        size = len(data_bytes)
        self._lib.wgpuQueueWriteBuffer(
            self._queue, buf_handle, 0,
            ctypes.c_char_p(data_bytes), size)

    def submit_dispatches(self, dispatches, readback=None):
        """Record and submit dispatches in a single compute pass.

        Args:
            dispatches: list of (pipeline, bind_group, (gx, gy, gz)) tuples
            readback: optional (gpu_buf_handle, size_bytes, numpy_dtype)
                      to copy-back after GPU finishes

        Returns:
            numpy array if readback requested, else None
        """
        lib = self._lib

        enc_desc = WGPUCommandEncoderDescriptor()
        enc_desc.nextInChain = None
        enc_desc.label = WGPUStringView.from_str("fast")
        encoder = lib.wgpuDeviceCreateCommandEncoder(
            self._device, ctypes.byref(enc_desc))

        pass_desc = WGPUComputePassDescriptor()
        pass_desc.nextInChain = None
        pass_desc.label = WGPUStringView.from_str("")
        pass_desc.timestampWrites = None

        compute_pass = lib.wgpuCommandEncoderBeginComputePass(
            encoder, ctypes.byref(pass_desc))

        for pipeline, bind_group, grid in dispatches:
            lib.wgpuComputePassEncoderSetPipeline(compute_pass, pipeline)
            lib.wgpuComputePassEncoderSetBindGroup(
                compute_pass, 0, bind_group, 0, None)
            gx = grid[0] if len(grid) > 0 else 1
            gy = grid[1] if len(grid) > 1 else 1
            gz = grid[2] if len(grid) > 2 else 1
            lib.wgpuComputePassEncoderDispatchWorkgroups(
                compute_pass, gx, gy, gz)

        lib.wgpuComputePassEncoderEnd(compute_pass)

        # Readback copy
        rb_buf = None
        if readback:
            gpu_handle, size, dtype = readback
            readback_usage = BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST
            rb_buf = self._get_or_create_buffer(
                "__fast_rb__", size, readback_usage)
            lib.wgpuCommandEncoderCopyBufferToBuffer(
                encoder, gpu_handle, 0, rb_buf, 0, size)

        # Finish + submit
        cb_desc = WGPUCommandBufferDescriptor()
        cb_desc.nextInChain = None
        cb_desc.label = WGPUStringView.from_str("")
        cmd_buf = lib.wgpuCommandEncoderFinish(
            encoder, ctypes.byref(cb_desc))
        cmd_bufs = (ctypes.c_void_p * 1)(cmd_buf)
        lib.wgpuQueueSubmit(
            self._queue, 1,
            ctypes.cast(cmd_bufs, ctypes.POINTER(WGPUCommandBuffer)))

        lib.wgpuComputePassEncoderRelease(compute_pass)
        lib.wgpuCommandEncoderRelease(encoder)
        lib.wgpuCommandBufferRelease(cmd_buf)

        if rb_buf:
            return self._map_and_read(rb_buf, size, dtype)
        return None

    def submit_dispatches_pipelined(self, layer_batches, readback=None,
                                    profiler=None, dispatch_names=None):
        """Submit dispatch batches with CPU/GPU pipelining.

        Groups consecutive batches into larger encoder submissions to reduce
        per-submit overhead while maintaining CPU/GPU overlap.

        Args:
            layer_batches: list of lists, each inner list is
                [(pipeline, bind_group, grid), ...] for one layer
            readback: optional (gpu_buf_handle, size_bytes, numpy_dtype)

        Returns:
            numpy array if readback requested, else None
        """
        lib = self._lib
        n_batches = len(layer_batches)
        GROUP_SIZE = 4  # layers per submit

        pass_desc = WGPUComputePassDescriptor()
        pass_desc.nextInChain = None
        pass_desc.label = WGPUStringView.from_str("")
        pass_desc.timestampWrites = None
        dispatch_idx = 0

        group_start = 0
        while group_start < n_batches:
            group_end = min(group_start + GROUP_SIZE, n_batches)
            is_last_group = (group_end == n_batches)

            enc_desc = WGPUCommandEncoderDescriptor()
            enc_desc.nextInChain = None
            enc_desc.label = WGPUStringView.from_str("pipe")
            encoder = lib.wgpuDeviceCreateCommandEncoder(
                self._device, ctypes.byref(enc_desc))

            compute_pass = lib.wgpuCommandEncoderBeginComputePass(
                encoder, ctypes.byref(pass_desc))

            for batch_idx in range(group_start, group_end):
                dispatches = layer_batches[batch_idx]

                for pipeline, bind_group, grid in dispatches:
                    # Optional per-dispatch GPU timestamps for profiler.
                    # We encode each dispatch in its own pass so timestampWrites
                    # can be attached per operation.
                    ts_ptr = None
                    if profiler and profiler.enabled and profiler.gpu_enabled:
                        if dispatch_names and dispatch_idx < len(dispatch_names):
                            dname = dispatch_names[dispatch_idx]
                        else:
                            dname = "fast_decode/dispatch"
                        b_idx, e_idx = profiler.allocate_gpu_timestamps(dname)
                        if b_idx >= 0:
                            ts_ptr = profiler.get_timestamp_writes_ptr(b_idx, e_idx)
                    dispatch_idx += 1

                    if ts_ptr is not None:
                        pass_desc.timestampWrites = ctypes.cast(ts_ptr, ctypes.c_void_p)
                        if compute_pass:
                            lib.wgpuComputePassEncoderEnd(compute_pass)
                            lib.wgpuComputePassEncoderRelease(compute_pass)
                        compute_pass = lib.wgpuCommandEncoderBeginComputePass(
                            encoder, ctypes.byref(pass_desc))
                    else:
                        pass_desc.timestampWrites = None

                    lib.wgpuComputePassEncoderSetPipeline(compute_pass, pipeline)
                    lib.wgpuComputePassEncoderSetBindGroup(
                        compute_pass, 0, bind_group, 0, None)
                    gx = grid[0] if len(grid) > 0 else 1
                    gy = grid[1] if len(grid) > 1 else 1
                    gz = grid[2] if len(grid) > 2 else 1
                    lib.wgpuComputePassEncoderDispatchWorkgroups(
                        compute_pass, gx, gy, gz)

            lib.wgpuComputePassEncoderEnd(compute_pass)
            lib.wgpuComputePassEncoderRelease(compute_pass)

            # Readback on last group
            if is_last_group and readback:
                gpu_handle, size, dtype = readback
                readback_usage = BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST
                rb_buf = self._get_or_create_buffer(
                    "__fast_rb__", size, readback_usage)
                lib.wgpuCommandEncoderCopyBufferToBuffer(
                    encoder, gpu_handle, 0, rb_buf, 0, size)

            cb_desc = WGPUCommandBufferDescriptor()
            cb_desc.nextInChain = None
            cb_desc.label = WGPUStringView.from_str("")
            cmd_buf = lib.wgpuCommandEncoderFinish(
                encoder, ctypes.byref(cb_desc))
            cmd_bufs = (ctypes.c_void_p * 1)(cmd_buf)
            lib.wgpuQueueSubmit(
                self._queue, 1,
                ctypes.cast(cmd_bufs, ctypes.POINTER(WGPUCommandBuffer)))

            lib.wgpuCommandEncoderRelease(encoder)
            lib.wgpuCommandBufferRelease(cmd_buf)

            group_start = group_end

        if readback:
            gpu_handle, size, dtype = readback
            readback_usage = BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST
            rb_buf = self._get_or_create_buffer(
                "__fast_rb__", size, readback_usage)
            return self._map_and_read(rb_buf, size, dtype)
        return None

    # -- GPU memory management ------------------------------------------------

    # Maximum bytes per wgpuQueueWriteBuffer call.
    # D3D12 staging heap is limited to 2GB; writes above this threshold
    # silently fail, producing a zero-filled GPU buffer.
    _MAX_WRITE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GiB

    def upload_to_gpu(self, data, name="tensor"):
        """Upload a numpy array to GPU memory and return a GPUBuffer handle.

        The returned GPUBuffer can be passed in the ``buffers`` dict of
        ``run_kernel()`` in place of a numpy array.  The runner will use the
        GPU buffer directly, skipping the per-call CPU→GPU upload.  This is
        ideal for model weights that are constant across inference steps.

        For buffers larger than 2 GiB, the upload is split into chunks
        because wgpuQueueWriteBuffer silently fails above the D3D12
        staging heap limit.
        """
        np_arr = np.ascontiguousarray(data)
        usage = (BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_SRC
                 | BUFFER_USAGE_COPY_DST)

        buf_desc = WGPUBufferDescriptor()
        buf_desc.nextInChain = None
        buf_desc.label = WGPUStringView.from_str(name)
        buf_desc.usage = usage
        buf_desc.size = np_arr.nbytes
        buf_desc.mappedAtCreation = 0

        buf = self._lib.wgpuDeviceCreateBuffer(
            self._device, ctypes.byref(buf_desc))
        if not buf:
            raise RuntimeError(f"Failed to create GPU buffer '{name}'")
        self._total_gpu_bytes += np_arr.nbytes
        self._gpu_alloc_count += 1

        total = np_arr.nbytes
        if total <= self._MAX_WRITE_SIZE:
            # Single write
            data_ptr = np_arr.ctypes.data_as(ctypes.c_void_p)
            self._lib.wgpuQueueWriteBuffer(
                self._queue, buf, 0, data_ptr, total)
        else:
            # Chunked write to stay under D3D12 staging limit
            chunk = self._MAX_WRITE_SIZE
            base = np_arr.ctypes.data
            offset = 0
            while offset < total:
                sz = min(chunk, total - offset)
                ptr = ctypes.c_void_p(base + offset)
                self._lib.wgpuQueueWriteBuffer(
                    self._queue, buf, offset, ptr, sz)
                offset += sz

        return GPUBuffer(self, buf, np_arr.nbytes,
                         np_arr.dtype, np_arr.shape)

    def readback(self, gpu_buffer, dtype=None):
        """Read a GPUBuffer back to a numpy array."""
        if dtype is None:
            dtype = gpu_buffer.dtype
        lib = self._lib
        size = gpu_buffer.size

        # Create temporary readback buffer
        rb_desc = WGPUBufferDescriptor()
        rb_desc.nextInChain = None
        rb_desc.label = WGPUStringView.from_str("readback")
        rb_desc.usage = BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST
        rb_desc.size = size
        rb_desc.mappedAtCreation = 0
        rb_buf = lib.wgpuDeviceCreateBuffer(
            self._device, ctypes.byref(rb_desc))

        # Encode copy + submit
        enc_desc = WGPUCommandEncoderDescriptor()
        enc_desc.nextInChain = None
        enc_desc.label = WGPUStringView.from_str("")
        encoder = lib.wgpuDeviceCreateCommandEncoder(
            self._device, ctypes.byref(enc_desc))
        lib.wgpuCommandEncoderCopyBufferToBuffer(
            encoder, gpu_buffer.handle, 0, rb_buf, 0, size)
        cb_desc = WGPUCommandBufferDescriptor()
        cb_desc.nextInChain = None
        cb_desc.label = WGPUStringView.from_str("")
        cmd_buf = lib.wgpuCommandEncoderFinish(
            encoder, ctypes.byref(cb_desc))
        cmd_bufs = (ctypes.c_void_p * 1)(cmd_buf)
        lib.wgpuQueueSubmit(
            self._queue, 1,
            ctypes.cast(cmd_bufs, ctypes.POINTER(WGPUCommandBuffer)))

        # Map and read
        map_done = [False]
        map_status = [0]

        @BufferMapCallback
        def on_map(status, message, ud1, ud2,
                   _md=map_done, _ms=map_status):
            _md[0] = True
            _ms[0] = status
        self._map_cb = on_map

        cb_info = WGPUBufferMapCallbackInfo()
        cb_info.nextInChain = None
        cb_info.mode = WGPUCallbackMode.WaitAnyOnly
        cb_info.callback = on_map
        cb_info.userdata1 = None
        cb_info.userdata2 = None
        future = lib.wgpuBufferMapAsync(
            rb_buf, MAP_MODE_READ, 0, size, cb_info)
        wait_info = WGPUFutureWaitInfo()
        wait_info.future = future
        wait_info.completed = 0
        lib.wgpuInstanceWaitAny(
            self._instance, 1, ctypes.byref(wait_info),
            ctypes.c_uint64(-1))

        if map_status[0] != WGPUMapAsyncStatus.Success:
            raise RuntimeError(
                f"Readback map failed: status={map_status[0]}")
        data_ptr = lib.wgpuBufferGetConstMappedRange(rb_buf, 0, size)
        result = np.ctypeslib.as_array(
            (ctypes.c_uint8 * size).from_address(data_ptr),
            shape=(size,)).copy()
        lib.wgpuBufferUnmap(rb_buf)
        lib.wgpuBufferDestroy(rb_buf)
        lib.wgpuBufferRelease(rb_buf)
        lib.wgpuCommandBufferRelease(cmd_buf)
        lib.wgpuCommandEncoderRelease(encoder)

        return np.frombuffer(result, dtype=dtype)

    def gpu_slice(self, gpu_buffer, offset_bytes, size_bytes, name="slice"):
        """Create a new GPU buffer containing a slice of an existing buffer.

        Uses GPU-to-GPU copy (CopyBufferToBuffer) — no CPU readback needed.
        This is essential for slicing large buffers (>2GB) that exceed the
        mapping limit for readback.

        Args:
            gpu_buffer: source GPUBuffer
            offset_bytes: byte offset into the source buffer
            size_bytes: number of bytes to copy
            name: label for the new buffer

        Returns:
            GPUBuffer containing the copied data
        """
        lib = self._lib
        usage = (BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_SRC
                 | BUFFER_USAGE_COPY_DST)

        buf_desc = WGPUBufferDescriptor()
        buf_desc.nextInChain = None
        buf_desc.label = WGPUStringView.from_str(name)
        buf_desc.usage = usage
        buf_desc.size = size_bytes
        buf_desc.mappedAtCreation = 0

        dst_buf = lib.wgpuDeviceCreateBuffer(
            self._device, ctypes.byref(buf_desc))
        if not dst_buf:
            raise RuntimeError(f"Failed to create GPU buffer '{name}'")

        # Encode GPU-to-GPU copy
        enc_desc = WGPUCommandEncoderDescriptor()
        enc_desc.nextInChain = None
        enc_desc.label = WGPUStringView.from_str("")
        encoder = lib.wgpuDeviceCreateCommandEncoder(
            self._device, ctypes.byref(enc_desc))
        lib.wgpuCommandEncoderCopyBufferToBuffer(
            encoder, gpu_buffer.handle, offset_bytes, dst_buf, 0, size_bytes)
        cb_desc = WGPUCommandBufferDescriptor()
        cb_desc.nextInChain = None
        cb_desc.label = WGPUStringView.from_str("")
        cmd_buf = lib.wgpuCommandEncoderFinish(
            encoder, ctypes.byref(cb_desc))
        cmd_bufs = (ctypes.c_void_p * 1)(cmd_buf)
        lib.wgpuQueueSubmit(
            self._queue, 1,
            ctypes.cast(cmd_bufs, ctypes.POINTER(WGPUCommandBuffer)))

        lib.wgpuCommandBufferRelease(cmd_buf)
        lib.wgpuCommandEncoderRelease(encoder)

        return GPUBuffer(self, dst_buf, size_bytes,
                         gpu_buffer.dtype, None)

    def _get_or_create_pipeline(self, wgsl_code, buffer_bindings, param_fields):
        """Get cached pipeline or create a new one.

        Returns (shader_module, pipeline, bg_layout, pipeline_layout).
        Pipelines are cached by WGSL code hash — the shader, layout, and
        pipeline are created once and reused across calls.
        """
        key = self._pipeline_cache_key(wgsl_code)
        with self._pipeline_cache_lock:
            cached = self._pipeline_cache.get(key)
        if cached is not None:
            return cached

        lib = self._lib

        # Create shader module
        wgsl_source = WGPUShaderSourceWGSL()
        wgsl_source.chain.next = None
        wgsl_source.chain.sType = WGPUSType.ShaderSourceWGSL
        wgsl_bytes = wgsl_code.encode("utf-8")
        wgsl_source.code = WGPUStringView(wgsl_bytes, len(wgsl_bytes))

        shader_desc = WGPUShaderModuleDescriptor()
        shader_desc.nextInChain = ctypes.cast(
            ctypes.pointer(wgsl_source.chain), ctypes.POINTER(WGPUChainedStruct)
        )
        shader_desc.label = WGPUStringView.from_str("triton_kernel")

        shader_module = lib.wgpuDeviceCreateShaderModule(
            self._device, ctypes.byref(shader_desc)
        )
        if not shader_module:
            raise RuntimeError("Failed to create WGSL shader module")

        # Create bind group layout
        n_bindings = len(buffer_bindings) + (1 if param_fields else 0)
        LayoutEntryArray = WGPUBindGroupLayoutEntry * n_bindings
        layout_entries = LayoutEntryArray()

        for i, bb in enumerate(buffer_bindings):
            ctypes.memset(ctypes.byref(layout_entries[i]), 0,
                         ctypes.sizeof(WGPUBindGroupLayoutEntry))
            layout_entries[i].binding = bb.binding
            layout_entries[i].visibility = SHADER_STAGE_COMPUTE
            if bb.access == 'read':
                layout_entries[i].buffer.type = WGPUBufferBindingType.ReadOnlyStorage
            else:
                layout_entries[i].buffer.type = WGPUBufferBindingType.Storage

        if param_fields:
            idx = len(buffer_bindings)
            ctypes.memset(ctypes.byref(layout_entries[idx]), 0,
                         ctypes.sizeof(WGPUBindGroupLayoutEntry))
            layout_entries[idx].binding = len(buffer_bindings)
            layout_entries[idx].visibility = SHADER_STAGE_COMPUTE
            layout_entries[idx].buffer.type = WGPUBufferBindingType.ReadOnlyStorage

        bg_layout_desc = WGPUBindGroupLayoutDescriptor()
        bg_layout_desc.nextInChain = None
        bg_layout_desc.label = WGPUStringView.from_str("")
        bg_layout_desc.entryCount = n_bindings
        bg_layout_desc.entries = layout_entries

        bg_layout = lib.wgpuDeviceCreateBindGroupLayout(
            self._device, ctypes.byref(bg_layout_desc)
        )

        # Create pipeline layout
        bg_layouts = (ctypes.c_void_p * 1)(bg_layout)
        pl_desc = WGPUPipelineLayoutDescriptor()
        pl_desc.nextInChain = None
        pl_desc.label = WGPUStringView.from_str("")
        pl_desc.bindGroupLayoutCount = 1
        pl_desc.bindGroupLayouts = ctypes.cast(bg_layouts, ctypes.POINTER(WGPUBindGroupLayout))
        pl_desc.immediateSize = 0

        pipeline_layout = lib.wgpuDeviceCreatePipelineLayout(
            self._device, ctypes.byref(pl_desc)
        )

        # Create compute pipeline
        cp_desc = WGPUComputePipelineDescriptor()
        cp_desc.nextInChain = None
        cp_desc.label = WGPUStringView.from_str("triton_pipeline")
        cp_desc.layout = pipeline_layout
        cp_desc.compute.nextInChain = None
        cp_desc.compute.module = shader_module
        cp_desc.compute.entryPoint = WGPUStringView.from_str("main")
        cp_desc.compute.constantCount = 0
        cp_desc.compute.constants = None

        if getattr(self, '_has_async_compute_pipeline', False):
            pipeline_holder = [None]
            pipeline_error = [None]

            @CreateComputePipelineAsyncCallback
            def on_pipeline(status, pipeline_obj, message, ud1, ud2,
                            _holder=pipeline_holder, _err=pipeline_error):
                if status == WGPUCreatePipelineAsyncStatus.Success:
                    _holder[0] = pipeline_obj
                else:
                    msg = ""
                    if message.data:
                        msg = message.data.decode("utf-8", errors="replace")
                    _err[0] = msg or f"status={status}"

            cb_info = WGPUCreateComputePipelineAsyncCallbackInfo()
            cb_info.nextInChain = None
            cb_info.mode = WGPUCallbackMode.WaitAnyOnly
            cb_info.callback = on_pipeline
            cb_info.userdata1 = None
            cb_info.userdata2 = None

            future = lib.wgpuDeviceCreateComputePipelineAsync(
                self._device, ctypes.byref(cp_desc), cb_info)
            wait_info = WGPUFutureWaitInfo()
            wait_info.future = future
            wait_info.completed = 0
            lib.wgpuInstanceWaitAny(
                self._instance, 1, ctypes.byref(wait_info),
                ctypes.c_uint64(-1))

            pipeline = pipeline_holder[0]
            if not pipeline:
                err = pipeline_error[0] or "unknown error"
                raise RuntimeError(f"Failed to create compute pipeline (async): {err}")
        else:
            pipeline = lib.wgpuDeviceCreateComputePipeline(
                self._device, ctypes.byref(cp_desc)
            )
        if not pipeline:
            raise RuntimeError("Failed to create compute pipeline")

        result = (shader_module, pipeline, bg_layout, pipeline_layout)
        with self._pipeline_cache_lock:
            existing = self._pipeline_cache.get(key)
            if existing is not None:
                # Another thread won the race; release duplicate resources.
                lib.wgpuComputePipelineRelease(pipeline)
                lib.wgpuShaderModuleRelease(shader_module)
                lib.wgpuBindGroupLayoutRelease(bg_layout)
                lib.wgpuPipelineLayoutRelease(pipeline_layout)
                return existing
            self._pipeline_cache[key] = result
        return result

    @staticmethod
    def _pipeline_cache_key(wgsl_code: str) -> str:
        import hashlib
        return hashlib.sha256(wgsl_code.encode()).hexdigest()

    @staticmethod
    def _round_to_size_class(size):
        """Round buffer size up to the next size class for pool reuse.

        Size classes balance fragmentation vs reuse:
          - Below 256 bytes: round to 256 (minimum WebGPU buffer)
          - 256 – 4KB: round to next multiple of 256
          - 4KB – 64KB: round to next power of 2
          - 64KB+: round to next power of 2

        This ensures buffers with similar sizes share pool slots,
        reducing total GPU allocations by ~50%.
        """
        if size <= 256:
            return 256
        if size <= 4096:
            return ((size + 255) // 256) * 256
        # Power of 2 rounding for larger buffers
        p = 1
        while p < size:
            p <<= 1
        return p

    def _pool_acquire(self, size, usage):
        """Acquire a buffer from the pool, or create a new one.

        Buffers are bucketed by rounded size class. If a free buffer
        of the right size class exists, it is reused (zero allocation
        cost). Otherwise a new buffer is created at the rounded size.

        Returns: (WGPUBuffer, actual_size)
        """
        rounded = self._round_to_size_class(size)
        free_list = self._pool_free.get(rounded)
        if free_list:
            buf = free_list.pop()
            self._pool_reuse += 1
            return buf, rounded

        # Create new buffer at the rounded size
        lib = self._lib
        buf_desc = WGPUBufferDescriptor()
        buf_desc.nextInChain = None
        buf_desc.label = WGPUStringView.from_str(f"pool_{rounded}")
        buf_desc.usage = usage
        buf_desc.size = rounded
        buf_desc.mappedAtCreation = 0
        buf = lib.wgpuDeviceCreateBuffer(self._device, ctypes.byref(buf_desc))
        if not buf:
            raise RuntimeError(f"Failed to create pool buffer (size={rounded})")
        self._pool_alloc += 1
        self._pool_bytes += rounded
        self._total_gpu_bytes += rounded
        self._gpu_alloc_count += 1
        return buf, rounded

    def _pool_release(self, buf, size):
        """Return a buffer to the pool for future reuse.

        The buffer is added to the free list for its size class.
        It is NOT destroyed — it stays allocated on GPU and can be
        immediately reused by the next _pool_acquire of the same class.
        """
        rounded = self._round_to_size_class(size)
        if rounded not in self._pool_free:
            self._pool_free[rounded] = []
        self._pool_free[rounded].append(buf)

    def _get_or_create_buffer(self, name, size, usage):
        """Get a cached GPU buffer or create a new one.

        Named buffers (weights, persistent data) are cached by exact
        (name, size, usage) key. Transient buffers (kernel I/O with
        names starting with '__') use the size-class memory pool for
        better reuse across different-sized operations.
        """
        key = (name, size, usage)
        if key in self._buffer_cache:
            return self._buffer_cache[key]

        # Use the pool for transient/internal buffers
        if name.startswith("__"):
            buf, actual_size = self._pool_acquire(size, usage)
            # Cache with rounded size so same name+size hits next time
            self._buffer_cache[key] = buf
            return buf

        lib = self._lib
        buf_desc = WGPUBufferDescriptor()
        buf_desc.nextInChain = None
        buf_desc.label = WGPUStringView.from_str(name)
        buf_desc.usage = usage
        buf_desc.size = size
        buf_desc.mappedAtCreation = 0

        buf = lib.wgpuDeviceCreateBuffer(self._device, ctypes.byref(buf_desc))
        if not buf:
            raise RuntimeError(f"Failed to create buffer '{name}'")
        self._buffer_cache[key] = buf
        self._total_gpu_bytes += size
        self._gpu_alloc_count += 1
        return buf

    def run_kernel(
        self,
        wgsl_code: str,
        buffer_bindings: list,
        param_fields: list,
        workgroup_size: int,
        grid: tuple,
        buffers: dict,
        scalars: dict = None,
        gpu_outputs: set = None,
        timestamp_writes_ptr=None,
    ) -> dict:
        """
        Execute a WGSL compute shader on the GPU.

        Pipelines and GPU buffers are cached across calls for efficiency.
        Only bind groups and command buffers are created per dispatch.

        Args:
            buffers: dict mapping binding name → numpy array *or* GPUBuffer.
                     GPUBuffer values bypass the CPU→GPU upload entirely.
            gpu_outputs: optional set of read_write buffer names that should
                         stay on GPU.  Those entries in the result dict will
                         be GPUBuffer objects instead of numpy arrays.
            timestamp_writes_ptr: optional ctypes pointer to
                         WGPUPassTimestampWrites for GPU profiling.
        """
        scalars = scalars or {}
        lib = self._lib

        # Get or create cached pipeline
        _, pipeline, bg_layout, _ = self._get_or_create_pipeline(
            wgsl_code, buffer_bindings, param_fields)

        return self._run_with_pipeline(
            pipeline, bg_layout, buffer_bindings, param_fields,
            workgroup_size, grid, buffers, scalars, gpu_outputs,
            timestamp_writes_ptr=timestamp_writes_ptr,
        )

    def _run_with_pipeline(self, pipeline, bg_layout, buffer_bindings,
                          param_fields, workgroup_size, grid, buffers,
                          scalars, gpu_outputs=None,
                          timestamp_writes_ptr=None):
        """Execute a compute dispatch using cached pipeline.

        GPU storage buffers are cached by (name, size) — only re-uploaded
        when contents change.  Readback buffers are also cached.
        Only bind groups, command encoders and command buffers are
        created per dispatch (these are lightweight).

        If *gpu_outputs* is given, the named read_write buffers are returned
        as GPUBuffer objects instead of being read back to numpy.
        """
        lib = self._lib
        batching = self.is_batching and gpu_outputs

        # 1. Allocate / reuse GPU storage buffers and upload data
        gpu_buffers = {}       # name -> WGPUBuffer
        gpu_buf_sizes = {}     # name -> size in bytes
        gpu_owned = {}         # name -> True if buffer is a fresh allocation for gpu_outputs

        storage_usage = (BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_SRC
                         | BUFFER_USAGE_COPY_DST)

        for bb in buffer_bindings:
            if bb.name in buffers:
                val = buffers[bb.name]
                if isinstance(val, GPUBuffer):
                    # Pre-uploaded GPU buffer — use directly, skip upload
                    gpu_buf = val.handle
                    buf_size = val.size
                    gpu_owned[bb.name] = False
                else:
                    np_arr = np.ascontiguousarray(val)
                    buf_size = np_arr.nbytes

                    if gpu_outputs and bb.name in gpu_outputs:
                        # Output stays on GPU — use toggle-cached buffer pool
                        # to avoid fresh allocation each call. Two buffers per
                        # (binding_name, size) alternate to prevent read-write
                        # aliasing between consecutive operations.
                        pool_key = f"__gpu_out_{bb.name}_{buf_size}"
                        toggle = self._gpu_out_toggles.get(pool_key, 0)
                        cache_name = f"{pool_key}_{toggle}"
                        self._gpu_out_toggles[pool_key] = 1 - toggle
                        gpu_buf = self._get_or_create_buffer(
                            cache_name, buf_size, storage_usage)
                        gpu_owned[bb.name] = False
                        # Skip uploading zeros — kernel overwrites entirely.
                    else:
                        gpu_buf = self._get_or_create_buffer(
                            bb.name, buf_size, storage_usage)
                        gpu_owned[bb.name] = False

                        # Upload data
                        data_ptr = np_arr.ctypes.data_as(ctypes.c_void_p)
                        lib.wgpuQueueWriteBuffer(
                            self._queue, gpu_buf, 0, data_ptr, np_arr.nbytes)
            else:
                buf_size = 16  # dummy
                gpu_buf = self._get_or_create_buffer(
                    bb.name, buf_size, storage_usage)
                gpu_owned[bb.name] = False

            gpu_buffers[bb.name] = gpu_buf
            gpu_buf_sizes[bb.name] = buf_size

        # 2. Params buffer (scalar arguments)
        params_buf = None
        params_size = 0
        if param_fields:
            params_data = bytearray()
            for pf in param_fields:
                val = scalars.get(pf.name, 0)
                fmt = WGSL_TYPE_TO_STRUCT_FMT.get(pf.wgsl_type, '<i')
                params_data.extend(struct.pack(fmt, val))
            while len(params_data) < 16:
                params_data.extend(b'\x00')
            params_size = len(params_data)

            # In batch mode each dispatch needs its own params buffer
            # because wgpuQueueWriteBuffer calls all complete before the
            # batch command buffer executes on the GPU.
            if batching:
                batch_idx = len(self._batch_cleanup) if hasattr(self, '_batch_cleanup') else 0
                params_name = f"__params_batch_{batch_idx}__"
            else:
                params_name = "__params__"

            params_buf = self._get_or_create_buffer(
                params_name, params_size,
                BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_DST)

            params_bytes = bytes(params_data)
            lib.wgpuQueueWriteBuffer(
                self._queue, params_buf, 0,
                ctypes.c_char_p(params_bytes), params_size
            )

        # 3. Create bind group (lightweight — not cached)
        n_bindings = len(buffer_bindings) + (1 if param_fields else 0)
        BindEntryArray = WGPUBindGroupEntry * n_bindings
        bind_entries = BindEntryArray()

        for i, bb in enumerate(buffer_bindings):
            ctypes.memset(ctypes.byref(bind_entries[i]), 0,
                         ctypes.sizeof(WGPUBindGroupEntry))
            bind_entries[i].binding = bb.binding
            bind_entries[i].buffer = gpu_buffers[bb.name]
            bind_entries[i].offset = 0
            bind_entries[i].size = gpu_buf_sizes[bb.name]

        if param_fields:
            idx = len(buffer_bindings)
            ctypes.memset(ctypes.byref(bind_entries[idx]), 0,
                         ctypes.sizeof(WGPUBindGroupEntry))
            bind_entries[idx].binding = len(buffer_bindings)
            bind_entries[idx].buffer = params_buf
            bind_entries[idx].offset = 0
            bind_entries[idx].size = params_size

        bg_desc = WGPUBindGroupDescriptor()
        bg_desc.nextInChain = None
        bg_desc.label = WGPUStringView.from_str("")
        bg_desc.layout = bg_layout
        bg_desc.entryCount = n_bindings
        bg_desc.entries = bind_entries

        bind_group = lib.wgpuDeviceCreateBindGroup(
            self._device, ctypes.byref(bg_desc)
        )

        # 4. Encode and submit compute pass
        #    In batch mode: use shared encoder, skip submit.
        #    In normal mode: create own encoder, submit + readback.

        if batching:
            encoder = self._batch_encoder
        else:
            enc_desc = WGPUCommandEncoderDescriptor()
            enc_desc.nextInChain = None
            enc_desc.label = WGPUStringView.from_str("")

            encoder = lib.wgpuDeviceCreateCommandEncoder(
                self._device, ctypes.byref(enc_desc)
            )

        pass_desc = WGPUComputePassDescriptor()
        pass_desc.nextInChain = None
        pass_desc.label = WGPUStringView.from_str("")
        if timestamp_writes_ptr is not None:
            # timestamp_writes_ptr is a ctypes.byref() to a WGPUPassTimestampWrites
            # We need the raw address as a c_void_p for the struct field
            pass_desc.timestampWrites = ctypes.cast(timestamp_writes_ptr,
                                                     ctypes.c_void_p)
        else:
            pass_desc.timestampWrites = None

        compute_pass = lib.wgpuCommandEncoderBeginComputePass(
            encoder, ctypes.byref(pass_desc)
        )
        lib.wgpuComputePassEncoderSetPipeline(compute_pass, pipeline)
        lib.wgpuComputePassEncoderSetBindGroup(compute_pass, 0, bind_group, 0, None)

        gx = grid[0] if len(grid) > 0 else 1
        gy = grid[1] if len(grid) > 1 else 1
        gz = grid[2] if len(grid) > 2 else 1

        lib.wgpuComputePassEncoderDispatchWorkgroups(compute_pass, gx, gy, gz)
        lib.wgpuComputePassEncoderEnd(compute_pass)

        # 5. Copy output buffers to readback buffers
        #    Buffers in gpu_outputs stay on GPU — no copy or readback.
        #    In batch mode: skip submit entirely, return GPUBuffer objects.
        if batching:
            # In batch mode: return GPUBuffer objects immediately
            # Accumulate pass/bind_group for later cleanup (after submit)
            if not hasattr(self, '_batch_cleanup'):
                self._batch_cleanup = []
            self._batch_cleanup.append((compute_pass, bind_group))
            results = {}
            for bb in buffer_bindings:
                if bb.name in buffers and bb.access == 'read_write':
                    if gpu_outputs and bb.name in gpu_outputs:
                        buf_size = gpu_buf_sizes[bb.name]
                        elem_type = bb.elem_type
                        np_dtype = WGSL_TYPE_TO_NUMPY.get(elem_type, np.float32)
                        results[bb.name] = GPUBuffer(
                            self, gpu_buffers[bb.name], buf_size, np_dtype,
                            owned=False)
            return results

        readback_buffers = {}
        readback_usage = BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST
        for bb in buffer_bindings:
            if bb.name in buffers and bb.access == 'read_write':
                if gpu_outputs and bb.name in gpu_outputs:
                    continue  # skip — will return GPUBuffer
                size = gpu_buf_sizes[bb.name]
                rb_buf = self._get_or_create_buffer(
                    f"__rb_{bb.name}__", size, readback_usage)
                readback_buffers[bb.name] = rb_buf
                lib.wgpuCommandEncoderCopyBufferToBuffer(
                    encoder, gpu_buffers[bb.name], 0, rb_buf, 0, size
                )

        # Finish and submit
        cb_desc = WGPUCommandBufferDescriptor()
        cb_desc.nextInChain = None
        cb_desc.label = WGPUStringView.from_str("")

        cmd_buf = lib.wgpuCommandEncoderFinish(encoder, ctypes.byref(cb_desc))
        cmd_bufs = (ctypes.c_void_p * 1)(cmd_buf)

        lib.wgpuQueueSubmit(
            self._queue, 1,
            ctypes.cast(cmd_bufs, ctypes.POINTER(WGPUCommandBuffer))
        )

        # 6. Read back results
        results = {}
        for bb_name, rb_buf in readback_buffers.items():
            size = gpu_buf_sizes[bb_name]

            map_done = [False]
            map_status = [0]

            @BufferMapCallback
            def on_map(status, message, ud1, ud2, _md=map_done, _ms=map_status):
                _md[0] = True
                _ms[0] = status

            self._map_cb = on_map

            cb_info = WGPUBufferMapCallbackInfo()
            cb_info.nextInChain = None
            cb_info.mode = WGPUCallbackMode.WaitAnyOnly
            cb_info.callback = on_map
            cb_info.userdata1 = None
            cb_info.userdata2 = None

            future = lib.wgpuBufferMapAsync(
                rb_buf, MAP_MODE_READ, 0, size, cb_info
            )

            wait_info = WGPUFutureWaitInfo()
            wait_info.future = future
            wait_info.completed = 0

            lib.wgpuInstanceWaitAny(
                self._instance, 1, ctypes.byref(wait_info),
                ctypes.c_uint64(-1)
            )

            if map_status[0] != WGPUMapAsyncStatus.Success:
                raise RuntimeError(f"Buffer map failed for '{bb_name}': status={map_status[0]}")

            data_ptr = lib.wgpuBufferGetConstMappedRange(rb_buf, 0, size)
            if not data_ptr:
                raise RuntimeError(f"Failed to get mapped range for '{bb_name}'")

            # Find elem type for this buffer
            elem_type = 'f32'
            for bb in buffer_bindings:
                if bb.name == bb_name:
                    elem_type = bb.elem_type
                    break
            np_dtype = WGSL_TYPE_TO_NUMPY.get(elem_type, np.float32)
            result_arr = np.ctypeslib.as_array(
                (ctypes.c_uint8 * size).from_address(data_ptr),
                shape=(size,)
            ).copy()  # copy before unmap
            results[bb_name] = np.frombuffer(result_arr, dtype=np_dtype)

            lib.wgpuBufferUnmap(rb_buf)

        # 7. Return GPU-resident outputs as GPUBuffer objects
        if gpu_outputs:
            for bb in buffer_bindings:
                if bb.name in gpu_outputs and bb.access == 'read_write':
                    elem_type = bb.elem_type
                    np_dtype = WGSL_TYPE_TO_NUMPY.get(elem_type, np.float32)
                    results[bb.name] = GPUBuffer(
                        self, gpu_buffers[bb.name],
                        gpu_buf_sizes[bb.name], np_dtype, owned=False)

        # 8. Cleanup (only per-call objects, not cached ones)
        lib.wgpuComputePassEncoderRelease(compute_pass)
        lib.wgpuCommandEncoderRelease(encoder)
        lib.wgpuCommandBufferRelease(cmd_buf)
        lib.wgpuBindGroupRelease(bind_group)

        return results

    def __del__(self):
        """Release Dawn resources."""
        lib = self._lib
        if lib is None:
            return
        try:
            if hasattr(self, '_pipeline_executor') and self._pipeline_executor:
                self._pipeline_executor.shutdown(wait=False)
            # Release cached buffers
            for buf in getattr(self, '_buffer_cache', {}).values():
                lib.wgpuBufferDestroy(buf)
                lib.wgpuBufferRelease(buf)
            # Release pooled free buffers
            for bufs in getattr(self, '_pool_free', {}).values():
                for buf in bufs:
                    lib.wgpuBufferDestroy(buf)
                    lib.wgpuBufferRelease(buf)
            # Release cached pipelines
            for sm, pipe, bgl, pl in getattr(self, '_pipeline_cache', {}).values():
                lib.wgpuComputePipelineRelease(pipe)
                lib.wgpuShaderModuleRelease(sm)
                lib.wgpuBindGroupLayoutRelease(bgl)
                lib.wgpuPipelineLayoutRelease(pl)
            if hasattr(self, '_queue') and self._queue:
                lib.wgpuQueueRelease(self._queue)
            if hasattr(self, '_device') and self._device:
                lib.wgpuDeviceRelease(self._device)
            if hasattr(self, '_adapter') and self._adapter:
                lib.wgpuAdapterRelease(self._adapter)
            if hasattr(self, '_instance') and self._instance:
                lib.wgpuInstanceRelease(self._instance)
        except Exception:
            pass


def run_triton_kernel_on_webgpu(
    compiled_kernel,
    grid: tuple,
    **kwargs,
) -> dict:
    """
    High-level API: Execute a compiled Triton WebGPU kernel on the GPU via Dawn.

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

    llir = compiled_kernel.asm.get('llir', '')
    if not llir:
        raise ValueError("Compiled kernel has no LLVM IR. Was it compiled for WebGPU?")

    metadata = compiled_kernel.metadata
    sig = {}
    if hasattr(compiled_kernel, 'signature'):
        sig = compiled_kernel.signature

    num_warps = metadata.get('num_warps', 4)
    warp_size = 32

    result = translate_llvm_to_wgsl(llir, sig, num_warps, warp_size)

    buffers_dict = {}
    scalars_dict = {}
    for name, val in kwargs.items():
        if isinstance(val, np.ndarray):
            buffers_dict[name] = val
        else:
            scalars_dict[name] = val

    runner = DawnRunner()
    return runner.run_kernel(
        wgsl_code=result.wgsl,
        buffer_bindings=result.buffer_bindings,
        param_fields=result.param_fields,
        workgroup_size=result.workgroup_size,
        grid=grid,
        buffers=buffers_dict,
        scalars=scalars_dict,
    )
