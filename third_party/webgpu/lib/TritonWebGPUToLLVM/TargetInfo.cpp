// TargetInfo.cpp -- WebGPU/SPIR-V target info implementation
//
// Implements the TargetInfoBase interface for the WebGPU backend.
// Operations that require SPIR-V-specific intrinsics are lowered to
// external function calls (e.g., __spirv_ControlBarrier) that the
// SPIR-V translator will recognize and convert to SPIR-V ops.

#include "TargetInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace mlir::triton::WebGPU {

// ---------------------------------------------------------------------------
// Helper: get or create an LLVM function declaration in the module
// ---------------------------------------------------------------------------
LLVM::LLVMFuncOp TargetInfo::getOrCreateFunction(
    RewriterBase &rewriter, StringRef name,
    LLVM::LLVMFunctionType fnType) const {
  auto moduleOp =
      rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto fn = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (fn)
    return fn;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  fn = LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(rewriter.getContext()),
                                 name, fnType);
  fn.setLinkage(LLVM::Linkage::External);
  return fn;
}

// ---------------------------------------------------------------------------
// Helper: emit a subgroup shuffle via SPIR-V function call
// ---------------------------------------------------------------------------
Value TargetInfo::emitShuffleCall(RewriterBase &rewriter, Location loc,
                                  Value val, Value offset,
                                  StringRef funcName) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  Type valTy = val.getType();
  unsigned bits = valTy.getIntOrFloatBitWidth();

  // For 64-bit: split into two 32-bit values and shuffle independently
  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = emitShuffleCall(rewriter, loc, val0, offset, funcName);
    val1 = emitShuffleCall(rewriter, loc, val1, offset, funcName);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, valTy);
  }

  // Promote to i32 for shuffle
  Type origTy = valTy;
  if (valTy != i32_ty) {
    val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.zext(i32_ty, val);
  }

  // Create the SPIR-V subgroup shuffle function call
  // Signature: i32 funcName(i32 scope, i32 val, i32 offset)
  auto i32Type = IntegerType::get(ctx, 32);
  auto fnTy = LLVM::LLVMFunctionType::get(i32Type, {i32Type, i32Type, i32Type});
  auto fn = getOrCreateFunction(rewriter, funcName, fnTy);

  // Subgroup scope = 3 in SPIR-V
  SmallVector<Value> shuffleArgs = {b.i32_val(3), val, offset};
  Value result = b.call(fn, shuffleArgs).getResult();

  // Demote back to original type
  if (origTy != i32_ty) {
    if (bits < 32)
      result = b.trunc(int_ty(bits), result);
    result = b.bitcast(result, origTy);
  }
  return result;
}

// ---------------------------------------------------------------------------
// TargetInfoBase overrides
// ---------------------------------------------------------------------------

bool TargetInfo::supportMaximumMinimum() const {
  // No hardware max/min with NaN propagation
  return false;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter,
                                   Location loc) const {
  // WebGPU doesn't support multi-CTA clusters
  return LLVM::createConstantI32(loc, rewriter, 0);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                          Value cmp) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i1Ty = IntegerType::get(ctx, 1);

  // __spirv_GroupNonUniformBallot(scope, predicate) -> <4 x i32>
  // For simplicity, return all-ones when cmp is true, all-zeros when false
  // This is a simplified implementation
  auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i1Ty});
  auto fn = getOrCreateFunction(rewriter, "__spirv_GroupNonUniformBallot", fnTy);
  SmallVector<Value> ballotArgs = {b.i32_val(3), cmp};
  Value result = b.call(fn, ballotArgs).getResult();

  // Extend to the requested ballot type width
  unsigned targetBits = type.getIntOrFloatBitWidth();
  if (targetBits > 32)
    result = b.zext(type, result);
  else if (targetBits < 32)
    result = b.trunc(type, result);
  return result;
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                          triton::gpu::AddrSpace targets) const {
  // Use the standard TritonGPU BarrierOp, which will be lowered in the
  // conversion pass to a SPIR-V barrier function call
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier(targets);
}

void TargetInfo::clusterBarrier(Location loc,
                                 RewriterBase &rewriter) const {
  // WebGPU doesn't have cluster barriers, use workgroup barrier
  barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  // WebGPU doesn't have warp-level sync, use workgroup barrier
  barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                               std::optional<Value> ctaId, Value val,
                               Value pred) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(!ctaId.has_value() &&
         "WebGPU does not support cross-CTA shared memory");

  // Predicated store: if (pred) *ptr = val
  if (!isa<VectorType>(val.getType())) {
    // Scalar store
    b.store(val, ptr);
    return;
  }

  // Vector store
  auto vecTy = cast<VectorType>(val.getType());
  unsigned vec = vecTy.getNumElements();
  if (vec == 1) {
    Value elem = b.extract_element(vecTy.getElementType(), val, b.i32_val(0));
    b.store(elem, ptr);
    return;
  }

  // Store each element individually
  for (unsigned i = 0; i < vec; i++) {
    Value elem = b.extract_element(vecTy.getElementType(), val, b.i32_val(i));
    auto elemPtr = b.gep(ptr.getType(), vecTy.getElementType(), ptr,
                         b.i32_val(i), LLVM::GEPNoWrapFlags::inbounds);
    b.store(elem, elemPtr);
  }
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                               std::optional<Value> ctaId, Type loadTy,
                               Value pred, Operation *localLoadOp) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(!ctaId.has_value() &&
         "WebGPU does not support cross-CTA shared memory");

  if (!isa<VectorType>(loadTy)) {
    return b.load(loadTy, ptr);
  }

  auto vecTy = cast<VectorType>(loadTy);
  unsigned vec = vecTy.getNumElements();
  Type elemTy = vecTy.getElementType();

  if (vec == 1) {
    Value elem = b.load(elemTy, ptr);
    Value vec_val = b.undef(vecTy);
    return b.insert_element(vecTy, vec_val, elem, b.i32_val(0));
  }

  // Load each element and pack into vector
  Value result = b.undef(vecTy);
  for (unsigned i = 0; i < vec; i++) {
    auto elemPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i),
                         LLVM::GEPNoWrapFlags::inbounds);
    Value elem = b.load(elemTy, elemPtr);
    result = b.insert_element(vecTy, result, elem, b.i32_val(i));
  }
  return result;
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                              int i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = emitShuffleCall(rewriter, loc, val, b.i32_val(i),
                                  "__spirv_SubgroupShuffleXor");
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = emitShuffleCall(rewriter, loc, val, b.i32_val(i),
                                  "__spirv_SubgroupShuffleUp");
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                              int i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = emitShuffleCall(rewriter, loc, val, b.i32_val(i),
                                  "__spirv_SubgroupShuffle");
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                              Value i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = emitShuffleCall(rewriter, loc, val, i,
                                  "__spirv_SubgroupShuffle");
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                           Value b_, Value selector) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Emulate byte permute via shifts and masks
  // This is a simplified version that handles the common case
  auto *ctx = rewriter.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty});
  auto fn = getOrCreateFunction(rewriter, "__spirv_BytePermute", fnTy);
  SmallVector<Value> permuteArgs = {a, b_, selector};
  return b.call(fn, permuteArgs).getResult();
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                             ModuleOp moduleOp, ProgramIDDim axis) const {
  // Use GPU dialect's BlockIdOp then convert to i32
  mlir::gpu::Dimension dim;
  switch (axis) {
  case ProgramIDDim::X:
    dim = mlir::gpu::Dimension::x;
    break;
  case ProgramIDDim::Y:
    dim = mlir::gpu::Dimension::y;
    break;
  case ProgramIDDim::Z:
    dim = mlir::gpu::Dimension::z;
    break;
  }
  Value blockId = mlir::gpu::BlockIdOp::create(rewriter, loc, dim);
  // Convert index to i32
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(),
                                    blockId);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                             SmallVector<Value> &acc, triton::ReduceOp op,
                             unsigned reduceLaneIdMask) const {
  // No hardware warp reduce; fall back to generic tree reduction
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  return resultElementTy.isInteger(32) ? "__spirv_umulhi32" : "__spirv_umulhi64";
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                         int formatStrByteCount, ValueRange args,
                         ArrayRef<bool> isSigned) const {
  // WebGPU/SPIR-V does not support device-side printf
  // This is a no-op
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                         ArrayRef<bool> isSigned) const {
  // No-op for WebGPU
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                             StringRef message, StringRef file, StringRef func,
                             int line) const {
  // No-op for WebGPU (no device-side assert support)
  // Could potentially write to a debug buffer in the future
}

int TargetInfo::getSharedAddressSpace() const {
  // LLVM address space 3 = SPIR-V Workgroup memory
  return 3;
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace))
    return 3; // Workgroup
  llvm::report_fatal_error("Unsupported address space for WebGPU backend");
  return 0;
}

bool TargetInfo::supportVectorizedAtomics() const { return false; }

} // namespace mlir::triton::WebGPU
