#pragma once

#include "mlir/Pass/Pass.h"

namespace ladder {

std::unique_ptr<mlir::Pass> createLowerONNXToLadderPass();

} // namespace ladder
