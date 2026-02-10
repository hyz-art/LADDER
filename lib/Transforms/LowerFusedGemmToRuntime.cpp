#include "ladder/IR/LadderOps.h"
#include "ladder/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LowerFusedGemmToRuntimePass
    : public PassWrapper<LowerFusedGemmToRuntimePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    auto getOrCreateFunc = [&](StringRef name, Type aType, Type bType, Type biasType, Type outType) -> func::FuncOp {
      if (auto func = module.lookupSymbol<func::FuncOp>(name))
        return func;

      auto i64 = builder.getI64Type();
      auto funcType = builder.getFunctionType({aType, bType, biasType, i64, i64, i64, i64, i64}, {outType});
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
      func->setAttr(SymbolTable::getVisibilityAttrName(), builder.getStringAttr("private"));
      return func;
    };

    auto precisionToI64 = [](StringRef prec) -> int64_t {
      if (prec == "fp16") return 1;
      if (prec == "fp8") return 2;
      if (prec == "int8") return 3;
      if (prec == "int4") return 4;
      return 0;
    };

    module.walk([&](ladder::GemmFusedOp op) {
      auto func = getOrCreateFunc("gemm_tiled_fused_tiles_cuda",
                                  op.getA().getType(), op.getB().getType(),
                                  op.getBias().getType(), op.getResult().getType());

      OpBuilder b(op);
      auto i64 = b.getI64Type();
      auto makeI64 = [&](int64_t v) {
        return b.create<arith::ConstantOp>(op.getLoc(), i64, b.getI64IntegerAttr(v));
      };

      int64_t tileM = op.getTileMAttr().getInt();
      int64_t tileN = op.getTileNAttr().getInt();
      int64_t tileK = op.getTileKAttr().getInt();
      int64_t fuseRelu = op.getFuseReluAttr().getInt();
      int64_t precision = precisionToI64(op.getPrecisionCAttr().getValue());

      auto call = b.create<func::CallOp>(op.getLoc(), func,
                                         ValueRange{op.getA(), op.getB(), op.getBias(),
                                                    makeI64(tileM), makeI64(tileN), makeI64(tileK),
                                                    makeI64(fuseRelu), makeI64(precision)});
      call->setAttrs(op->getAttrs());
      op.getResult().replaceAllUsesWith(call.getResult(0));
      op.erase();
    });
  }
};

} // namespace

namespace ladder {

std::unique_ptr<mlir::Pass> createLowerFusedGemmToRuntimePass() {
  return std::make_unique<LowerFusedGemmToRuntimePass>();
}

} // namespace ladder
