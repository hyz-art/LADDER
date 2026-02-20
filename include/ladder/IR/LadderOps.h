#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "ladder/IR/LadderDialect.h"

#define GET_OP_CLASSES
#include "LadderOps.h.inc"
