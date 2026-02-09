#include "ladder/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LowerTileToRuntimePass
    : public PassWrapper<LowerTileToRuntimePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    auto getOrCreateFunc = [&](StringRef name, Type inputType) -> func::FuncOp {
      if (auto func = module.lookupSymbol<func::FuncOp>(name))
        return func;

      auto i64 = builder.getI64Type();
      auto funcType = builder.getFunctionType({inputType, i64, i64, i64, i64}, {inputType});
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
      return func;
    };

    module.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name != "ladder.tile.copy" && name != "ladder.tile.prefetch" &&
          name != "ladder.tile.async_copy")
        return;

      if (op->getNumOperands() < 1 || op->getNumResults() < 1)
        return;

      auto input = op->getOperand(0);
      auto resultType = op->getResult(0).getType();

      StringRef callee;
      if (name == "ladder.tile.copy")
        callee = "ladder_cuda_tile_copy";
      else if (name == "ladder.tile.prefetch")
        callee = "ladder_cuda_tile_prefetch";
      else
        callee = "ladder_cuda_tile_async_copy";

      auto func = getOrCreateFunc(callee, input.getType());

      int64_t direction = 0;
      int64_t level = 0;
      if (auto dirAttr = op->getAttrOfType<StringAttr>("direction")) {
        if (dirAttr.getValue() == "local_to_global" || dirAttr.getValue() == "l2g")
          direction = 1;
      }
      if (auto levelAttr = op->getAttrOfType<IntegerAttr>("level"))
        level = levelAttr.getInt();

      OpBuilder b(op);
      auto i64 = b.getI64Type();
      auto dirC = b.create<arith::ConstantOp>(op->getLoc(), i64, b.getI64IntegerAttr(direction));
      auto lvlC = b.create<arith::ConstantOp>(op->getLoc(), i64, b.getI64IntegerAttr(level));
      auto streamC = b.create<arith::ConstantOp>(op->getLoc(), i64, b.getI64IntegerAttr(0));
      auto eventC = b.create<arith::ConstantOp>(op->getLoc(), i64, b.getI64IntegerAttr(0));
      auto call = b.create<func::CallOp>(op->getLoc(), func,
                                         ValueRange{input, dirC, lvlC, streamC, eventC});

      op->getResult(0).replaceAllUsesWith(call.getResult(0));
      op->erase();
    });
  }
};

} // namespace

namespace ladder {

std::unique_ptr<mlir::Pass> createLowerTileToRuntimePass() {
  return std::make_unique<LowerTileToRuntimePass>();
}

} // namespace ladder
