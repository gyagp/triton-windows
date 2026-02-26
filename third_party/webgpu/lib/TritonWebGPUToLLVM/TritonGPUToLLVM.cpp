// TritonGPUToLLVM.cpp -- WebGPU/SPIR-V backend conversion pass
//
// Converts TritonGPU IR to LLVM IR targeting SPIR-V.
// Reuses generic conversion patterns from lib/Conversion/TritonGPUToLLVM/
// and adds WebGPU-specific patterns for load/store, barriers, and GPU ops.

#include "TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace mgpu = ::mlir::gpu;  // Disambiguate from triton::gpu

// ============================================================================
// Helper function declarations
// ============================================================================

static LLVM::LLVMFuncOp
getOrCreateFuncDecl(RewriterBase &rewriter, ModuleOp mod, StringRef name,
                    LLVM::LLVMFunctionType fnType) {
  auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (fn)
    return fn;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  fn = LLVM::LLVMFuncOp::create(
      rewriter, UnknownLoc::get(rewriter.getContext()), name, fnType);
  fn.setLinkage(LLVM::Linkage::External);
  return fn;
}

// ============================================================================
// WebGPU-specific load/store patterns
// ============================================================================

namespace {

// Convert triton::LoadOp to standard LLVM loads
struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
  using ConvertOpToLLVMPattern<triton::LoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Get the result type
    auto resultTy = dyn_cast<RankedTensorType>(op.getType());
    if (!resultTy) {
      // Scalar load
      Value ptr = adaptor.getPtr();
      Value result = b.load(typeConverter->convertType(op.getType()), ptr);
      if (op.getMask()) {
        Value mask = adaptor.getMask();
        Value other = adaptor.getOther();
        if (!other)
          other = b.undef(result.getType());
        result = b.select(mask, result, other);
      }
      rewriter.replaceOp(op, result);
      return success();
    }

    Type elemTy = typeConverter->convertType(resultTy.getElementType());
    unsigned numElems = ttg::getTotalElemsPerThread(resultTy);

    SmallVector<Value> ptrElems =
        unpackLLElements(loc, adaptor.getPtr(), rewriter);
    SmallVector<Value> maskElems;
    SmallVector<Value> otherElems;

    if (op.getMask()) {
      maskElems = unpackLLElements(loc, adaptor.getMask(), rewriter);
    }
    if (op.getOther()) {
      otherElems = unpackLLElements(loc, adaptor.getOther(), rewriter);
    }

    SmallVector<Value> resultElems;
    for (unsigned i = 0; i < numElems; i++) {
      Value ptr = ptrElems[i];
      Value loaded = b.load(elemTy, ptr);

      if (!maskElems.empty()) {
        Value mask = maskElems[i];
        Value other = otherElems.empty() ? b.undef(elemTy) : otherElems[i];
        loaded = b.select(mask, loaded, other);
      }
      resultElems.push_back(loaded);
    }

    Value result = packLLElements(loc, getTypeConverter(), resultElems, rewriter,
                                  resultTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Convert triton::StoreOp to standard LLVM stores
struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp> {
  using ConvertOpToLLVMPattern<triton::StoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto valueTy = dyn_cast<RankedTensorType>(op.getValue().getType());
    if (!valueTy) {
      // Scalar store
      Value ptr = adaptor.getPtr();
      Value val = adaptor.getValue();
      b.store(val, ptr);
      rewriter.eraseOp(op);
      return success();
    }

    unsigned numElems = ttg::getTotalElemsPerThread(valueTy);

    SmallVector<Value> ptrElems =
        unpackLLElements(loc, adaptor.getPtr(), rewriter);
    SmallVector<Value> valElems =
        unpackLLElements(loc, adaptor.getValue(), rewriter);
    SmallVector<Value> maskElems;

    if (op.getMask()) {
      maskElems = unpackLLElements(loc, adaptor.getMask(), rewriter);
    }

    for (unsigned i = 0; i < numElems; i++) {
      Value ptr = ptrElems[i];
      Value val = valElems[i];

      if (!maskElems.empty()) {
        // Predicated store: use LLVM conditional branch
        // For simplicity, we use unconditional store + mask check
        // This assumes the pointer is valid even when mask is false
        // TODO: Use llvm.masked.store for proper masking
        Value mask = maskElems[i];
        Value zero = b.undef(val.getType());
        // Only store when mask is true by selecting value
        // Still stores but writes undef to avoid side effects
        // A proper implementation would skip the store entirely
        val = b.select(mask, val, zero);
      }
      b.store(val, ptr);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Convert triton::AtomicRMWOp to LLVM atomicrmw
struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
  using ConvertOpToLLVMPattern<triton::AtomicRMWOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Map Triton's atomic RMW kind to LLVM's
    auto mapAtomicOp = [](triton::RMWOp rmwOp) -> LLVM::AtomicBinOp {
      switch (rmwOp) {
      case triton::RMWOp::AND:
        return LLVM::AtomicBinOp::_and;
      case triton::RMWOp::OR:
        return LLVM::AtomicBinOp::_or;
      case triton::RMWOp::XOR:
        return LLVM::AtomicBinOp::_xor;
      case triton::RMWOp::ADD:
        return LLVM::AtomicBinOp::add;
      case triton::RMWOp::FADD:
        return LLVM::AtomicBinOp::fadd;
      case triton::RMWOp::MAX:
        return LLVM::AtomicBinOp::max;
      case triton::RMWOp::MIN:
        return LLVM::AtomicBinOp::min;
      case triton::RMWOp::UMAX:
        return LLVM::AtomicBinOp::umax;
      case triton::RMWOp::UMIN:
        return LLVM::AtomicBinOp::umin;
      case triton::RMWOp::XCHG:
        return LLVM::AtomicBinOp::xchg;
      }
      llvm_unreachable("Unsupported atomic RMW op");
    };

    auto resultTy = dyn_cast<RankedTensorType>(op.getType());
    if (!resultTy) {
      // Scalar atomic
      Value ptr = adaptor.getPtr();
      Value val = adaptor.getVal();
      auto result = LLVM::AtomicRMWOp::create(
          rewriter, loc, mapAtomicOp(op.getAtomicRmwOp()), ptr, val,
          LLVM::AtomicOrdering::monotonic);
      rewriter.replaceOp(op, result.getResult());
      return success();
    }

    unsigned numElems = ttg::getTotalElemsPerThread(resultTy);
    SmallVector<Value> ptrElems =
        unpackLLElements(loc, adaptor.getPtr(), rewriter);
    SmallVector<Value> valElems =
        unpackLLElements(loc, adaptor.getVal(), rewriter);
    SmallVector<Value> maskElems;
    if (op.getMask())
      maskElems = unpackLLElements(loc, adaptor.getMask(), rewriter);

    Type elemTy =
        typeConverter->convertType(resultTy.getElementType());
    SmallVector<Value> resultElems;

    for (unsigned i = 0; i < numElems; i++) {
      Value ptr = ptrElems[i];
      Value val = valElems[i];
      auto atomicOp = LLVM::AtomicRMWOp::create(
          rewriter, loc, mapAtomicOp(op.getAtomicRmwOp()), ptr, val,
          LLVM::AtomicOrdering::monotonic);
      Value result = atomicOp.getResult();
      if (!maskElems.empty()) {
        result = b.select(maskElems[i], result, b.undef(elemTy));
      }
      resultElems.push_back(result);
    }

    Value result = packLLElements(loc, getTypeConverter(), resultElems, rewriter,
                                  resultTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Convert triton::AtomicCASOp to LLVM cmpxchg
struct AtomicCASOPConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp> {
  using ConvertOpToLLVMPattern<triton::AtomicCASOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto resultTy = dyn_cast<RankedTensorType>(op.getType());
    if (!resultTy) {
      Value ptr = adaptor.getPtr();
      Value cmp = adaptor.getCmp();
      Value val = adaptor.getVal();
      auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
          rewriter, loc, ptr, cmp, val, LLVM::AtomicOrdering::monotonic,
          LLVM::AtomicOrdering::monotonic);
      // extract_val result type should be the element type (cmp's type), not
      // the struct type {T, i1} returned by cmpxchg.
      Value result = b.extract_val(cmp.getType(), cmpxchg, 0);
      rewriter.replaceOp(op, result);
      return success();
    }

    unsigned numElems = ttg::getTotalElemsPerThread(resultTy);
    SmallVector<Value> ptrElems =
        unpackLLElements(loc, adaptor.getPtr(), rewriter);
    SmallVector<Value> cmpElems =
        unpackLLElements(loc, adaptor.getCmp(), rewriter);
    SmallVector<Value> valElems =
        unpackLLElements(loc, adaptor.getVal(), rewriter);

    Type elemTy =
        typeConverter->convertType(resultTy.getElementType());
    SmallVector<Value> resultElems;

    for (unsigned i = 0; i < numElems; i++) {
      auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
          rewriter, loc, ptrElems[i], cmpElems[i], valElems[i],
          LLVM::AtomicOrdering::monotonic, LLVM::AtomicOrdering::monotonic);
      resultElems.push_back(b.extract_val(elemTy, cmpxchg, 0));
    }

    Value result = packLLElements(loc, getTypeConverter(), resultElems, rewriter,
                                  resultTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Convert triton::gpu::WarpIdOp to threadId / warpSize
struct WarpIdOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::WarpIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    if (triton::gpu::lookupNumWarps(op) == 1) {
      rewriter.replaceOp(op, b.i32_val(0));
      return success();
    }

    // Get threadId.x as i32
    Value tid = mgpu::ThreadIdOp::create(rewriter, loc,
                                          mgpu::Dimension::x);
    tid = arith::IndexCastOp::create(rewriter, loc, i32_ty, tid);

    if (std::optional<int> startId =
            getWarpGroupStartThreadId(rewriter.getInsertionBlock()))
      tid = LLVM::SubOp::create(rewriter, loc, tid, b.i32_val(*startId));

    int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    Value warpId = b.udiv(tid, b.i32_val(threadsPerWarp));
    rewriter.replaceOp(op, warpId);
    return success();
  }
};

// Convert triton::DotOp using FMA (no tensor cores)
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return convertFMADot(op, adaptor, getTypeConverter(), rewriter);
  }
};

// Convert triton::GetNumProgramsOp to gpu::GridDimOp
// (backend-specific: not in the generic populateSPMDOpToLLVMPattern)
struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    mgpu::Dimension dim;
    switch (op.getAxis()) {
    case ProgramIDDim::X:
      dim = mgpu::Dimension::x;
      break;
    case ProgramIDDim::Y:
      dim = mgpu::Dimension::y;
      break;
    case ProgramIDDim::Z:
      dim = mgpu::Dimension::z;
      break;
    }
    Value gridDim = mgpu::GridDimOp::create(rewriter, loc, dim);
    Value result = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(), gridDim);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Convert mgpu::BarrierOp to a SPIR-V-compatible barrier function call
struct GPUBarrierOpConversion
    : public ConvertOpToLLVMPattern<mgpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mgpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // __spirv_ControlBarrier(execution_scope, memory_scope, memory_semantics)
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty, i32Ty});
    auto fn = getOrCreateFuncDecl(rewriter, moduleOp,
                                   "__spirv_ControlBarrier", fnTy);

    // Scope::Workgroup = 2
    // MemorySemantics::AcquireRelease | MemorySemantics::WorkgroupMemory
    // = 0x8 | 0x100 = 0x108
    SmallVector<Value> barrierArgs = {b.i32_val(2), b.i32_val(2), b.i32_val(0x108)};
    b.call(fn, barrierArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

// Convert mgpu::BlockIdOp to SPIR-V workgroup ID function call
struct GPUBlockIdOpConversion
    : public ConvertOpToLLVMPattern<mgpu::BlockIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mgpu::BlockIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});

    auto fn = getOrCreateFuncDecl(rewriter, moduleOp,
                                   "__spirv_BuiltInWorkgroupId", fnTy);

    int dimIdx = 0;
    switch (op.getDimension()) {
    case mgpu::Dimension::x:
      dimIdx = 0;
      break;
    case mgpu::Dimension::y:
      dimIdx = 1;
      break;
    case mgpu::Dimension::z:
      dimIdx = 2;
      break;
    }

    SmallVector<Value> dimArgs = {b.i32_val(dimIdx)};
    Value result = b.call(fn, dimArgs).getResult();
    // BlockIdOp returns index type; convert to index
    Value indexResult = arith::IndexCastOp::create(rewriter, loc,
                                                    rewriter.getIndexType(),
                                                    result);
    rewriter.replaceOp(op, indexResult);
    return success();
  }
};

// Convert mgpu::ThreadIdOp to SPIR-V local invocation ID
struct GPUThreadIdOpConversion
    : public ConvertOpToLLVMPattern<mgpu::ThreadIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mgpu::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});

    auto fn = getOrCreateFuncDecl(rewriter, moduleOp,
                                   "__spirv_BuiltInLocalInvocationId", fnTy);

    int dimIdx = 0;
    switch (op.getDimension()) {
    case mgpu::Dimension::x:
      dimIdx = 0;
      break;
    case mgpu::Dimension::y:
      dimIdx = 1;
      break;
    case mgpu::Dimension::z:
      dimIdx = 2;
      break;
    }

    SmallVector<Value> dimArgs = {b.i32_val(dimIdx)};
    Value result = b.call(fn, dimArgs).getResult();
    Value indexResult = arith::IndexCastOp::create(rewriter, loc,
                                                    rewriter.getIndexType(),
                                                    result);
    rewriter.replaceOp(op, indexResult);
    return success();
  }
};

// Convert mgpu::GridDimOp
struct GPUGridDimOpConversion
    : public ConvertOpToLLVMPattern<mgpu::GridDimOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mgpu::GridDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});

    auto fn = getOrCreateFuncDecl(rewriter, moduleOp,
                                   "__spirv_BuiltInNumWorkgroups", fnTy);

    int dimIdx = 0;
    switch (op.getDimension()) {
    case mgpu::Dimension::x:
      dimIdx = 0;
      break;
    case mgpu::Dimension::y:
      dimIdx = 1;
      break;
    case mgpu::Dimension::z:
      dimIdx = 2;
      break;
    }

    SmallVector<Value> dimArgs = {b.i32_val(dimIdx)};
    Value result = b.call(fn, dimArgs).getResult();
    Value indexResult = arith::IndexCastOp::create(rewriter, loc,
                                                    rewriter.getIndexType(),
                                                    result);
    rewriter.replaceOp(op, indexResult);
    return success();
  }
};

// Convert mgpu::BlockDimOp
struct GPUBlockDimOpConversion
    : public ConvertOpToLLVMPattern<mgpu::BlockDimOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mgpu::BlockDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});

    auto fn = getOrCreateFuncDecl(rewriter, moduleOp,
                                   "__spirv_BuiltInWorkgroupSize", fnTy);

    int dimIdx = 0;
    switch (op.getDimension()) {
    case mgpu::Dimension::x:
      dimIdx = 0;
      break;
    case mgpu::Dimension::y:
      dimIdx = 1;
      break;
    case mgpu::Dimension::z:
      dimIdx = 2;
      break;
    }

    SmallVector<Value> dimArgs = {b.i32_val(dimIdx)};
    Value result = b.call(fn, dimArgs).getResult();
    Value indexResult = arith::IndexCastOp::create(rewriter, loc,
                                                    rewriter.getIndexType(),
                                                    result);
    rewriter.replaceOp(op, indexResult);
    return success();
  }
};

// ============================================================================
// Conversion target for WebGPU: LLVM is legal, Triton/GPU is illegal
// ============================================================================

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mgpu::GPUDialect>();
    addLegalOp<UnrealizedConversionCastOp>();
    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
    addDynamicallyLegalOp<triton::gpu::GlobalScratchAllocOp>(
        [](triton::gpu::GlobalScratchAllocOp op) {
          return op.getBackend() != "default";
        });
  }
};

// ============================================================================
// Main conversion pass
// ============================================================================

struct ConvertTritonWebGPUToLLVM
    : public OperationPass<ModuleOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTritonWebGPUToLLVM)

  ConvertTritonWebGPUToLLVM()
      : OperationPass<ModuleOp>(TypeID::get<ConvertTritonWebGPUToLLVM>()) {}

  ConvertTritonWebGPUToLLVM(const ConvertTritonWebGPUToLLVM &other)
      : OperationPass<ModuleOp>(other) {}

  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<ConvertTritonWebGPUToLLVM>(*this);
  }

  StringRef getArgument() const override {
    return "convert-triton-webgpu-to-llvm";
  }
  StringRef getDescription() const override {
    return "Convert TritonGPU IR to LLVM IR for WebGPU/SPIR-V target";
  }
  StringRef getName() const override { return "ConvertTritonWebGPUToLLVM"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, mlir::arith::ArithDialect,
                    mlir::math::MathDialect, mgpu::GPUDialect,
                    mlir::scf::SCFDialect, triton::TritonDialect,
                    ttg::TritonGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    WebGPU::TargetInfo targetInfo;

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(mod,
                                triton::defaultAllocationAnalysisScratchSizeFn);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    // Lower functions first
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
      if (failed(applyPartialConversion(mod, funcTarget,
                                        std::move(funcPatterns))))
        return signalPassFailure();
    }

    // Initialize shared memory
    initSharedMemory(typeConverter);

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    int webgpuBenefit = benefit + 1;

    // --- WebGPU-specific patterns (higher priority) ---
    patterns.add<LoadOpConversion>(typeConverter, webgpuBenefit);
    patterns.add<StoreOpConversion>(typeConverter, webgpuBenefit);
    patterns.add<AtomicRMWOpConversion>(typeConverter, webgpuBenefit);
    patterns.add<AtomicCASOPConversion>(typeConverter, webgpuBenefit);
    patterns.add<DotOpConversion>(typeConverter, webgpuBenefit);
    patterns.add<GetNumProgramsOpConversion>(typeConverter, webgpuBenefit);
    patterns.add<WarpIdOpConversion>(typeConverter, webgpuBenefit);

    // GPU dialect conversions (barrier, block/thread IDs)
    patterns.add<GPUBarrierOpConversion>(typeConverter, webgpuBenefit);
    patterns.add<GPUBlockIdOpConversion>(typeConverter, webgpuBenefit);
    patterns.add<GPUThreadIdOpConversion>(typeConverter, webgpuBenefit);
    patterns.add<GPUGridDimOpConversion>(typeConverter, webgpuBenefit);
    patterns.add<GPUBlockDimOpConversion>(typeConverter, webgpuBenefit);

    // --- Generic patterns (lower priority) ---
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                 patterns, benefit);
    mlir::triton::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

    // Float arith ops are NOT in the common populateElementwiseOpToLLVMPatterns;
    // each backend must register them via ElementwiseOpConversion so that
    // struct-packed tensor operands are unpacked to scalars before creating
    // the LLVM op. Without these, the stock ArithToLLVM patterns match and
    // produce illegal llvm.fadd/fmul/etc. with struct-typed operands.
    {
      using namespace mlir::triton::gpu;
#define POPULATE_FLOAT_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit)
      POPULATE_FLOAT_OP(arith::AddFOp, LLVM::FAddOp);
      POPULATE_FLOAT_OP(arith::SubFOp, LLVM::FSubOp);
      POPULATE_FLOAT_OP(arith::MulFOp, LLVM::FMulOp);
      POPULATE_FLOAT_OP(arith::DivFOp, LLVM::FDivOp);
      POPULATE_FLOAT_OP(arith::ExtFOp, LLVM::FPExtOp);
      POPULATE_FLOAT_OP(arith::TruncFOp, LLVM::FPTruncOp);
      POPULATE_FLOAT_OP(arith::FPToSIOp, LLVM::FPToSIOp);
      POPULATE_FLOAT_OP(arith::SIToFPOp, LLVM::SIToFPOp);
#undef POPULATE_FLOAT_OP
    }

    mlir::triton::populateMinMaxFOpToLLVMPattern(typeConverter, patterns,
                                                 axisInfoAnalysis, false,
                                                 benefit);
    mlir::triton::populateClampFOpToLLVMPattern(
        typeConverter, patterns, axisInfoAnalysis, targetInfo,
        patternBenefitClampOptimizedPattern);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateInstrumentationToLLVMPatterns(typeConverter,
                                                        patterns);

    // Standard MLIR conversions
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    // Apply conversion
    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    // Lower CF ops separately
    {
      TritonLLVMFunctionConversionTarget cfTarget(*context);
      cfTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
        return op->getDialect() !=
               context->getLoadedDialect<cf::ControlFlowDialect>();
      });
      RewritePatternSet cfPatterns(context);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            cfPatterns);
      if (failed(applyPartialConversion(mod, cfTarget, std::move(cfPatterns))))
        return signalPassFailure();
    }

    fixUpLoopAnnotation(mod);
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Dynamic shared allocation: array size 0 with external linkage
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Address space 3 = SPIR-V Workgroup memory
        static_cast<unsigned>(3));
  }
};

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

namespace mlir::triton::WebGPU {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonWebGPUToLLVMPass() {
  return std::make_unique<ConvertTritonWebGPUToLLVM>();
}

} // namespace mlir::triton::WebGPU
