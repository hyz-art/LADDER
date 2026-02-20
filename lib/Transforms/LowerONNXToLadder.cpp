#include "ladder/IR/LadderOps.h"
#include "ladder/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace {

static std::string getPrecisionString(Type type) {
  if (!type)
    return "unknown";
  if (type.isF16()) return "fp16";
  if (type.isF32()) return "fp32";
  if (type.isF64()) return "fp64";
  if (type.isBF16()) return "bf16";
  if (auto intTy = type.dyn_cast<IntegerType>()) {
    if (intTy.getWidth() == 8 && intTy.isUnsigned()) return "uint8";
    if (intTy.getWidth() == 8) return "int8";
    if (intTy.getWidth() == 16 && intTy.isUnsigned()) return "uint16";
    if (intTy.getWidth() == 16) return "int16";
    if (intTy.getWidth() == 32 && intTy.isUnsigned()) return "uint32";
    if (intTy.getWidth() == 32) return "int32";
  }
  return "unknown";
}

static std::string getTensorElemPrecision(Type type) {
  if (auto ranked = type.dyn_cast<RankedTensorType>())
    return getPrecisionString(ranked.getElementType());
  if (auto unranked = type.dyn_cast<UnrankedTensorType>())
    return getPrecisionString(unranked.getElementType());
  return getPrecisionString(type);
}

static float extractScale(Value value) {
  if (!value)
    return 1.0f;
  if (auto *defOp = value.getDefiningOp()) {
    if (auto cst = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto dense = cst.getValue().dyn_cast<DenseElementsAttr>()) {
        if (dense.getNumElements() > 0) {
          if (dense.getElementType().isF32())
            return (*dense.getValues<float>().begin());
          if (dense.getElementType().isF64())
            return static_cast<float>(*dense.getValues<double>().begin());
        }
      }
    }
  }
  return 1.0f;
}

static int64_t extractZeroPoint(Value value) {
  if (!value)
    return 0;
  if (auto *defOp = value.getDefiningOp()) {
    if (auto cst = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto dense = cst.getValue().dyn_cast<DenseElementsAttr>()) {
        if (dense.getNumElements() > 0) {
          if (auto intTy = dense.getElementType().dyn_cast<IntegerType>()) {
            if (intTy.getWidth() <= 64) {
              return (*dense.getValues<APInt>().begin()).getSExtValue();
            }
          }
        }
      }
    }
  }
  return 0;
}

static float extractCalibration(Operation *op, float scale) {
  if (!op)
    return scale;
  if (auto attr = op->getAttrOfType<FloatAttr>("calibration"))
    return static_cast<float>(attr.getValueAsDouble());
  return scale;
}

struct LowerONNXToLadderPass
    : public PassWrapper<LowerONNXToLadderPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "onnx.Add") {
        if (op->getNumOperands() < 2 || op->getNumResults() < 1)
          return;

        OpBuilder builder(op);
        auto resultType = op->getResult(0).getType();
        auto add = builder.create<ladder::AddOp>(
            op->getLoc(), resultType, op->getOperand(0), op->getOperand(1));
        op->getResult(0).replaceAllUsesWith(add.getResult());
        op->erase();
        return;
      }

      if (name == "onnx.Relu") {
        if (op->getNumOperands() < 1 || op->getNumResults() < 1)
          return;

        OpBuilder builder(op);
        auto resultType = op->getResult(0).getType();
        auto map = builder.create<ladder::MapOp>(
            op->getLoc(), resultType, op->getOperand(0),
            builder.getStringAttr("relu"), builder.getArrayAttr({}));
        op->getResult(0).replaceAllUsesWith(map.getResult());
        op->erase();
        return;
      }
      if (name == "onnx.QuantizeLinear" || name == "onnx.DequantizeLinear") {
        if (op->getNumOperands() < 2 || op->getNumResults() < 1)
          return;

        OpBuilder builder(op);
        auto resultType = op->getResult(0).getType();
        float scale = extractScale(op->getOperand(1));
        int64_t zeroPoint = (op->getNumOperands() >= 3) ? extractZeroPoint(op->getOperand(2)) : 0;
        float calibration = extractCalibration(op, scale);
        auto precision = getTensorElemPrecision(resultType);

        if (name == "onnx.QuantizeLinear") {
          auto qop = builder.create<ladder::QuantizeOp>(
              op->getLoc(), resultType, op->getOperand(0),
              builder.getF32FloatAttr(scale), builder.getI64IntegerAttr(zeroPoint),
              builder.getF32FloatAttr(calibration), builder.getStringAttr(precision));
          op->getResult(0).replaceAllUsesWith(qop.getResult());
          op->erase();
          return;
        }

        auto dqop = builder.create<ladder::DequantizeOp>(
            op->getLoc(), resultType, op->getOperand(0),
            builder.getF32FloatAttr(scale), builder.getI64IntegerAttr(zeroPoint),
            builder.getF32FloatAttr(calibration), builder.getStringAttr(precision));
        op->getResult(0).replaceAllUsesWith(dqop.getResult());
        op->erase();
        return;
      }

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

      Value a = op->getOperand(0);
      Value b = op->getOperand(1);

      std::string precisionA = getTensorElemPrecision(a.getType());
      std::string precisionB = getTensorElemPrecision(b.getType());
      std::string precisionC = getTensorElemPrecision(resultType);

      // Type sinking for onnx.Cast around GEMM inputs.
      if (auto *castOp = a.getDefiningOp()) {
        if (castOp->getName().getStringRef() == "onnx.Cast" && castOp->hasOneUse()) {
          precisionA = getTensorElemPrecision(castOp->getResult(0).getType());
          a = castOp->getOperand(0);
        }
      }
      if (auto *castOp = b.getDefiningOp()) {
        if (castOp->getName().getStringRef() == "onnx.Cast" && castOp->hasOneUse()) {
          precisionB = getTensorElemPrecision(castOp->getResult(0).getType());
          b = castOp->getOperand(0);
        }
      }

      std::string accumulate = "fp32";
      std::string accumulatePartial = "fp32";
      int64_t accumulatePartialBlock = 0;
      int64_t tileK = 16;
      if (precisionA == "fp64" || precisionB == "fp64" || precisionC == "fp64") {
        accumulate = "fp64";
        accumulatePartial = "fp64";
        accumulatePartialBlock = 0;
      } else if (precisionA == "bf16" || precisionB == "bf16" ||
                 precisionA == "fp16" || precisionB == "fp16") {
        accumulate = "fp32";
        accumulatePartial = "fp32";
        if (K > 0 && tileK > 0)
          accumulatePartialBlock = (K + tileK - 1) / tileK;
      } else {
        accumulate = precisionC;
        accumulatePartial = precisionC;
        accumulatePartialBlock = 0;
      }

      auto gemm = builder.create<ladder::GemmOp>(
          op->getLoc(), resultType,
          a, b,
          builder.getI64IntegerAttr(M), builder.getI64IntegerAttr(N), builder.getI64IntegerAttr(K),
          builder.getI64IntegerAttr(64), builder.getI64IntegerAttr(64), builder.getI64IntegerAttr(tileK),
          builder.getStringAttr(precisionA), builder.getStringAttr(precisionB), builder.getStringAttr(precisionC),
          builder.getStringAttr(accumulate), builder.getStringAttr(accumulatePartial),
          builder.getI64IntegerAttr(accumulatePartialBlock),
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
