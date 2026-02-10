#pragma once

#include "mlir/Pass/Pass.h"

namespace ladder {

std::unique_ptr<mlir::Pass> createLowerONNXToLadderPass();
std::unique_ptr<mlir::Pass> createLowerTileToRuntimePass();
std::unique_ptr<mlir::Pass> createFuseLadderOpsPass();
std::unique_ptr<mlir::Pass> createLowerFusedGemmToRuntimePass();
std::unique_ptr<mlir::Pass> createLowerGemmToRuntimePass();

} // namespace ladder
