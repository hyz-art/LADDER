#include "ladder/IR/LadderOps.h"
#include "ladder/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static ladder::GemmFusedOp createFusedGemm(OpBuilder &b, ladder::GemmOp gemm,
                                           Value bias, bool fuseRelu) {
  return b.create<ladder::GemmFusedOp>(
      gemm.getLoc(), gemm.getResult().getType(),
      gemm.getOperand(0), gemm.getOperand(1), bias,
      gemm.getMAttr(), gemm.getNAttr(), gemm.getKAttr(),
      gemm.getTileMAttr(), gemm.getTileNAttr(), gemm.getTileKAttr(),
      gemm.getPrecisionAAttr(), gemm.getPrecisionBAttr(), gemm.getPrecisionCAttr(),
      gemm.getAccumulateAttr(), gemm.getAccumulatePartialAttr(),
      gemm.getAccumulatePartialBlockAttr(),
      gemm.getInputScaleAttr(), gemm.getOutputScaleAttr(),
      gemm.getImplAttr(), b.getI64IntegerAttr(fuseRelu ? 1 : 0));
}

struct FuseLadderOpsPass
    : public PassWrapper<FuseLadderOpsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Fuse gemm + add (+ relu)
    module.walk([&](ladder::MapOp mapOp) {
      if (mapOp.getMapFnAttr().getValue() != "relu")
        return;
      auto addOp = mapOp.getInput().getDefiningOp<ladder::AddOp>();
      if (!addOp || !addOp->hasOneUse())
        return;

      ladder::GemmOp gemm;
      Value bias;
      if (auto g = addOp.getLhs().getDefiningOp<ladder::GemmOp>()) {
        gemm = g;
        bias = addOp.getRhs();
      } else if (auto g = addOp.getRhs().getDefiningOp<ladder::GemmOp>()) {
        gemm = g;
        bias = addOp.getLhs();
      } else {
        return;
      }
      if (!gemm->hasOneUse())
        return;

      OpBuilder b(mapOp);
      auto fused = createFusedGemm(b, gemm, bias, true);
      mapOp.getResult().replaceAllUsesWith(fused.getResult());
      mapOp.erase();
      addOp.erase();
      gemm.erase();
    });

    // Fuse gemm + add (no relu)
    module.walk([&](ladder::AddOp addOp) {
      if (!addOp->hasOneUse())
        return;
      if (auto mapUser = dyn_cast<ladder::MapOp>(*addOp->user_begin())) {
        if (mapUser.getMapFnAttr().getValue() == "relu")
          return; // handled above
      }

      ladder::GemmOp gemm;
      Value bias;
      if (auto g = addOp.getLhs().getDefiningOp<ladder::GemmOp>()) {
        gemm = g;
        bias = addOp.getRhs();
      } else if (auto g = addOp.getRhs().getDefiningOp<ladder::GemmOp>()) {
        gemm = g;
        bias = addOp.getLhs();
      } else {
        return;
      }
      if (!gemm->hasOneUse())
        return;

      OpBuilder b(addOp);
      auto fused = createFusedGemm(b, gemm, bias, false);
      addOp.getResult().replaceAllUsesWith(fused.getResult());
      addOp.erase();
      gemm.erase();
    });

    // Fuse gemm + relu (no bias) by toggling fuse_relu.
    module.walk([&](ladder::MapOp mapOp) {
      if (mapOp.getMapFnAttr().getValue() != "relu")
        return;
      auto gemm = mapOp.getInput().getDefiningOp<ladder::GemmOp>();
      if (!gemm || !gemm->hasOneUse())
        return;

      OpBuilder b(mapOp);
      auto newGemm = b.create<ladder::GemmOp>(
          gemm.getLoc(), gemm.getResult().getType(),
          gemm.getOperand(0), gemm.getOperand(1),
          gemm.getMAttr(), gemm.getNAttr(), gemm.getKAttr(),
          gemm.getTileMAttr(), gemm.getTileNAttr(), gemm.getTileKAttr(),
          gemm.getPrecisionAAttr(), gemm.getPrecisionBAttr(), gemm.getPrecisionCAttr(),
          gemm.getAccumulateAttr(), gemm.getAccumulatePartialAttr(),
          gemm.getAccumulatePartialBlockAttr(),
          gemm.getInputScaleAttr(), gemm.getOutputScaleAttr(),
          gemm.getImplAttr(), b.getI64IntegerAttr(1));

      mapOp.getResult().replaceAllUsesWith(newGemm.getResult());
      mapOp.erase();
      gemm.erase();
    });
  }
};

} // namespace

namespace ladder {

std::unique_ptr<mlir::Pass> createFuseLadderOpsPass() {
  return std::make_unique<FuseLadderOpsPass>();
}

} // namespace ladder
