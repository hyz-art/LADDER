#include "ladder/IR/LadderDialect.h"
#include "ladder/IR/LadderOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace ladder;

void LadderDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LadderOps.cpp.inc"
      >();
}
