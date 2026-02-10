#include "ladder/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LowerTileToRuntimePass
    : public PassWrapper<LowerTileToRuntimePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    auto getOrCreateFunc = [&](StringRef name, Type inputType, int numI64) -> func::FuncOp {
      if (auto func = module.lookupSymbol<func::FuncOp>(name))
        return func;

      auto i64 = builder.getI64Type();
      SmallVector<Type, 8> inputs;
      inputs.push_back(inputType);
      for (int i = 0; i < numI64; ++i)
        inputs.push_back(i64);
      auto funcType = builder.getFunctionType(inputs, {inputType});
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
      func->setAttr(SymbolTable::getVisibilityAttrName(), builder.getStringAttr("private"));
      return func;
    };

    auto getArrayI64 = [](ArrayAttr arr, int idx, int64_t def) -> int64_t {
      if (!arr || idx >= static_cast<int>(arr.size()))
        return def;
      if (auto intAttr = arr[idx].dyn_cast<IntegerAttr>())
        return intAttr.getInt();
      return def;
    };

    module.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      bool isCopy = name == "ladder.tile.copy";
      bool isPrefetch = name == "ladder.tile.prefetch";
      bool isAsyncCopy = name == "ladder.tile.async_copy";
      bool isExtract = name == "ladder.tile.extract";
      bool isMap = name == "ladder.tile.map";
      bool isPad = name == "ladder.tile.pad";
      bool isTransform = name == "ladder.tile.transform";
      if (!isCopy && !isPrefetch && !isAsyncCopy && !isExtract && !isMap && !isPad && !isTransform)
        return;

      if (op->getNumOperands() < 1 || op->getNumResults() < 1)
        return;

      auto input = op->getOperand(0);
      auto resultType = op->getResult(0).getType();

      StringRef callee;
      int numI64 = 0;
      if (isCopy) {
        callee = "ladder_cuda_tile_copy";
        numI64 = 7;
      } else if (isPrefetch) {
        callee = "ladder_cuda_tile_prefetch";
        numI64 = 7;
      } else if (isAsyncCopy) {
        callee = "ladder_cuda_tile_async_copy";
        numI64 = 7;
      } else if (isExtract) {
        callee = "ladder_cuda_tile_extract";
        numI64 = 11;
      } else if (isMap) {
        callee = "ladder_cuda_tile_map";
        numI64 = 9;
      } else if (isPad) {
        callee = "ladder_cuda_tile_pad";
        numI64 = 12;
      } else if (isTransform) {
        callee = "ladder_cuda_tile_transform";
        numI64 = 9;
      }

      auto func = getOrCreateFunc(callee, input.getType(), numI64);

      int64_t direction = 0;
      int64_t level = 0;
      int64_t layoutKind = 0;
      int64_t layoutParam0 = 0;
      int64_t layoutParam1 = 0;
      if (auto dirAttr = op->getAttrOfType<StringAttr>("direction")) {
        if (dirAttr.getValue() == "local_to_global" || dirAttr.getValue() == "l2g")
          direction = 1;
      }
      if (auto levelAttr = op->getAttrOfType<IntegerAttr>("level"))
        level = levelAttr.getInt();
      if (auto layoutAttr = op->getAttrOfType<StringAttr>("layout")) {
        auto v = layoutAttr.getValue();
        if (v == "blocked") layoutKind = 1;
        else if (v == "packed") layoutKind = 2;
        else layoutKind = 0;
      }
      if (auto paramsAttr = op->getAttrOfType<ArrayAttr>("layout_params")) {
        if (paramsAttr.size() > 0) {
          if (auto intAttr = paramsAttr[0].dyn_cast<IntegerAttr>())
            layoutParam0 = intAttr.getInt();
        }
        if (paramsAttr.size() > 1) {
          if (auto intAttr = paramsAttr[1].dyn_cast<IntegerAttr>())
            layoutParam1 = intAttr.getInt();
        }
      }

      OpBuilder b(op);
      auto i64 = b.getI64Type();
      auto makeI64 = [&](int64_t v) {
        return b.create<arith::ConstantOp>(op->getLoc(), i64, b.getI64IntegerAttr(v));
      };

      SmallVector<Value, 16> args;
      args.push_back(input);

      if (isCopy || isAsyncCopy) {
        args.push_back(makeI64(direction));
        args.push_back(makeI64(0)); // level
      } else if (isPrefetch) {
        args.push_back(makeI64(0)); // direction
        args.push_back(makeI64(level));
      } else if (isExtract) {
        auto offsets = op->getAttrOfType<ArrayAttr>("offsets");
        auto sizes = op->getAttrOfType<ArrayAttr>("sizes");
        auto strides = op->getAttrOfType<ArrayAttr>("strides");
        args.push_back(makeI64(getArrayI64(offsets, 0, 0)));
        args.push_back(makeI64(getArrayI64(offsets, 1, 0)));
        args.push_back(makeI64(getArrayI64(sizes, 0, 0)));
        args.push_back(makeI64(getArrayI64(sizes, 1, 0)));
        args.push_back(makeI64(getArrayI64(strides, 0, 1)));
        args.push_back(makeI64(getArrayI64(strides, 1, 1)));
      } else if (isMap) {
        int64_t mapFn = 0;
        if (auto mapAttr = op->getAttrOfType<StringAttr>("map_fn")) {
          if (mapAttr.getValue() == "relu") mapFn = 1;
        }
        auto axis = op->getAttrOfType<ArrayAttr>("axis_map");
        args.push_back(makeI64(mapFn));
        args.push_back(makeI64(getArrayI64(axis, 0, 0)));
        args.push_back(makeI64(getArrayI64(axis, 1, 1)));
      } else if (isPad) {
        auto padLow = op->getAttrOfType<ArrayAttr>("pad_low");
        auto padHigh = op->getAttrOfType<ArrayAttr>("pad_high");
        auto padInner = op->getAttrOfType<ArrayAttr>("pad_inner");
        int64_t padValueBits = 0;
        if (auto pv = op->getAttrOfType<FloatAttr>("pad_value")) {
          auto bits = pv.getValue().bitcastToAPInt();
          padValueBits = static_cast<int64_t>(bits.getZExtValue());
        }
        args.push_back(makeI64(getArrayI64(padLow, 0, 0)));
        args.push_back(makeI64(getArrayI64(padLow, 1, 0)));
        args.push_back(makeI64(getArrayI64(padHigh, 0, 0)));
        args.push_back(makeI64(getArrayI64(padHigh, 1, 0)));
        args.push_back(makeI64(getArrayI64(padInner, 0, 0)));
        args.push_back(makeI64(getArrayI64(padInner, 1, 0)));
        args.push_back(makeI64(padValueBits));
      } else if (isTransform) {
        int64_t kind = 0;
        if (auto kindAttr = op->getAttrOfType<StringAttr>("transform_kind")) {
          if (kindAttr.getValue() == "transpose") kind = 1;
        }
        auto params = op->getAttrOfType<ArrayAttr>("params");
        args.push_back(makeI64(kind));
        args.push_back(makeI64(getArrayI64(params, 0, 0)));
        args.push_back(makeI64(getArrayI64(params, 1, 0)));
      }

      auto streamC = makeI64(0);
      auto eventC = makeI64(0);
      auto layoutC = makeI64(layoutKind);
      auto layoutP0 = makeI64(layoutParam0);
      auto layoutP1 = makeI64(layoutParam1);

      if (isCopy || isPrefetch || isAsyncCopy) {
        args.push_back(streamC);
        args.push_back(eventC);
        args.push_back(layoutC);
        args.push_back(layoutP0);
        args.push_back(layoutP1);
      } else if (isExtract || isMap || isPad || isTransform) {
        args.push_back(streamC);
        args.push_back(eventC);
        args.push_back(layoutC);
        args.push_back(layoutP0);
        args.push_back(layoutP1);
      }

      auto call = b.create<func::CallOp>(op->getLoc(), func, args);

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
