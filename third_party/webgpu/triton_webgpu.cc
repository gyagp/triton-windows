// triton_webgpu.cc -- Pybind11 module for WebGPU backend
//
// Provides:
//   webgpu.load_dialects(ctx)       -- no-op for now (no custom dialects)
//   webgpu.set_spv_target_triple(mod) -- set SPIR-V target triple on LLVM module
//   webgpu.translate_llvmir_to_spirv(llvmIR) -> (bytes, name)
//   webgpu.passes.ttgpuir.add_to_llvmir(pm) -- TritonGPU→LLVM conversion pass
//
// The SPIR-V translation reuses the same approach as Intel's XPU backend:
//   1. Parse LLVM IR string
//   2. Set spir64 target triple + data layout
//   3. Use LLVM's SPIR-V serialization (via MLIR libs)

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <string>

namespace py = pybind11;
using ret = py::return_value_policy;

// Forward declaration for the WebGPU TritonGPU→LLVM conversion pass
namespace mlir::triton::WebGPU {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertTritonWebGPUToLLVMPass();
} // namespace mlir::triton::WebGPU

// Find kernel functions in the LLVM module.
// For SPIR-V, kernels use CallingConv::SPIR_KERNEL.
// For standard LLVM IR (from Triton), they are just regular functions with
// external linkage whose names don't start with '_' or 'llvm.'.
static std::string findKernelName(llvm::Module &M) {
  // First, try SPIR_KERNEL calling convention
  for (auto &F : M.functions()) {
    if (F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      return F.getName().str();
    }
  }
  // Fallback: find the first non-intrinsic, non-declaration function
  for (auto &F : M.functions()) {
    if (!F.isDeclaration() && !F.isIntrinsic() &&
        F.getLinkage() == llvm::GlobalValue::ExternalLinkage) {
      return F.getName().str();
    }
  }
  return "";
}

// Set the SPIR-V target triple and data layout on an LLVM module.
static void setSpvTargetTriple(llvm::Module *mod) {
  std::string triple = "spir64-unknown-unknown";
  std::string layout =
      "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:"
      "256-v256:256-v512:512-v1024:1024-n8:16:32:64";
  mod->setTargetTriple(llvm::Triple(triple));
  mod->setDataLayout(layout);
}

// Translate LLVM IR to SPIR-V binary.
// This produces a SPIR-V module from the LLVM IR by:
//   1. Parsing the LLVM IR
//   2. Setting SPIR-V target triple
//   3. Converting calling conventions to SPIR_KERNEL
//   4. Serializing to SPIR-V bitcode
//
// NOTE: Full SPIR-V translation requires either:
//   a) LLVM's experimental SPIR-V backend (not in bundled LLVM), or
//   b) The llvm-spirv translator, or
//   c) MLIR's GPU->SPIR-V pipeline
// For now, we output LLVM bitcode with SPIR-V triple as an intermediate
// format. The actual SPIR-V translation will be handled by an external
// tool (e.g., llvm-spirv) or by Dawn's Tint shader compiler.
static std::tuple<py::object, std::string>
translateToSpirvBitcode(const std::string &llvmIR) {
  std::string name;
  std::string spirvBitcode;
  {
    py::gil_scoped_release allow_threads;

    llvm::LLVMContext context;
    std::unique_ptr<llvm::MemoryBuffer> buffer =
        llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
    llvm::SMDiagnostic error;
    std::unique_ptr<llvm::Module> module =
        llvm::parseIR(buffer->getMemBufferRef(), error, context);

    if (!module) {
      llvm::report_fatal_error("failed to parse IR: " + error.getMessage() +
                               " lineno: " +
                               std::to_string(error.getLineNo()));
    }

    // Set SPIR-V target
    setSpvTargetTriple(module.get());

    // Convert kernel calling conventions to SPIR_KERNEL
    for (auto &F : module->functions()) {
      if (!F.isDeclaration() && !F.isIntrinsic() &&
          F.getLinkage() == llvm::GlobalValue::ExternalLinkage) {
        F.setCallingConv(llvm::CallingConv::SPIR_KERNEL);
      }
    }

    // Find kernel name
    name = findKernelName(*module);

    // Serialize module to LLVM bitcode (with SPIR-V triple)
    llvm::SmallVector<char, 0> buffer_vec;
    llvm::raw_svector_ostream os(buffer_vec);
    llvm::WriteBitcodeToFile(*module, os);
    spirvBitcode.assign(buffer_vec.begin(), buffer_vec.end());
  }
  return std::make_tuple(py::bytes(spirvBitcode), name);
}

void init_triton_webgpu(py::module &&m) {
  // Load dialects (no custom WebGPU dialects for now)
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    // WebGPU backend doesn't define custom MLIR dialects yet.
    // It reuses the standard Triton/TritonGPU dialects.
    context.loadAllAvailableDialects();
  });

  // Set SPIR-V target triple on an LLVM module
  m.def("set_spv_target_triple", [](llvm::Module *mod) {
    setSpvTargetTriple(mod);
  });

  // Translate LLVM IR to SPIR-V bitcode
  m.def(
      "translate_to_spirv",
      [](const std::string &llvmIR)
          -> std::tuple<py::object, std::string> {
        return translateToSpirvBitcode(llvmIR);
      },
      ret::take_ownership);

  // Submodule for passes
  auto passes = m.def_submodule("passes");
  auto ttgpuir = passes.def_submodule("ttgpuir");

  // Register the TritonGPU → LLVM conversion pass for WebGPU/SPIR-V
  ttgpuir.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::WebGPU::createConvertTritonWebGPUToLLVMPass());
  });
}
