#include "ladder/IR/LadderDialect.h"
#include "ladder/IR/LadderOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace ladder;

// Dialect constructor/registration (generated)
#include "LadderDialect.cpp.inc"

void LadderDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LadderOps.cpp.inc"
      >();
}
