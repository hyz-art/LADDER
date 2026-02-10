#include "ladder/IR/LadderOps.h"
#include "ladder/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LowerGemmToRuntimePass
    : public PassWrapper<LowerGemmToRuntimePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    auto getOrCreateFunc = [&](StringRef name, Type aType, Type bType, Type outType) -> func::FuncOp {
      if (auto func = module.lookupSymbol<func::FuncOp>(name))
        return func;

      auto funcType = builder.getFunctionType({aType, bType}, {outType});
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
      func->setAttr(SymbolTable::getVisibilityAttrName(), builder.getStringAttr("private"));
      return func;
    };

    module.walk([&](ladder::GemmOp op) {
      auto func = getOrCreateFunc("gemm_tiled_fused_tiles",
                                  op.getA().getType(), op.getB().getType(), op.getResult().getType());

      OpBuilder b(op);
      auto call = b.create<func::CallOp>(op.getLoc(), func, ValueRange{op.getA(), op.getB()});
      call->setAttrs(op->getAttrs());
      op.getResult().replaceAllUsesWith(call.getResult(0));
      op.erase();
    });
  }
};

} // namespace

namespace ladder {

std::unique_ptr<mlir::Pass> createLowerGemmToRuntimePass() {
  return std::make_unique<LowerGemmToRuntimePass>();
}

} // namespace ladder
