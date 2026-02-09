#include "ladder/IR/LadderOps.h"
#include "ladder/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LowerONNXToLadderPass
    : public PassWrapper<LowerONNXToLadderPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name != "onnx.MatMul" && name != "onnx.Gemm")
        return;

      if (op->getNumOperands() < 2 || op->getNumResults() < 1)
        return;

      OpBuilder builder(op);
      auto resultType = op->getResult(0).getType();

      // Infer M/N/K if possible
      int64_t M = -1, N = -1, K = -1;
      if (auto aType = op->getOperand(0).getType().dyn_cast<RankedTensorType>()) {
        if (aType.getRank() == 2) {
          M = aType.getShape()[0];
          K = aType.getShape()[1];
        }
      }
      if (auto bType = op->getOperand(1).getType().dyn_cast<RankedTensorType>()) {
        if (bType.getRank() == 2) {
          if (K == -1) K = bType.getShape()[0];
          N = bType.getShape()[1];
        }
      }

      auto gemm = builder.create<ladder::GemmOp>(
          op->getLoc(), resultType,
          op->getOperand(0), op->getOperand(1),
          builder.getI64IntegerAttr(M), builder.getI64IntegerAttr(N), builder.getI64IntegerAttr(K),
          builder.getI64IntegerAttr(64), builder.getI64IntegerAttr(64), builder.getI64IntegerAttr(16),
          builder.getStringAttr("fp16"), builder.getStringAttr("fp32"),
          builder.getF32FloatAttr(1.0f), builder.getF32FloatAttr(1.0f),
          builder.getStringAttr("ttile"), builder.getI64IntegerAttr(0));

      op->getResult(0).replaceAllUsesWith(gemm.getResult());
      op->erase();
    });
  }
};

} // namespace

namespace ladder {

std::unique_ptr<mlir::Pass> createLowerONNXToLadderPass() {
  return std::make_unique<LowerONNXToLadderPass>();
}

} // namespace ladder
