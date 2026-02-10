#include "ladder/Transforms/Passes.h"

namespace ladder {

std::unique_ptr<mlir::Pass> createLowerTileToRuntimePass();
std::unique_ptr<mlir::Pass> createFuseLadderOpsPass();
std::unique_ptr<mlir::Pass> createLowerFusedGemmToRuntimePass();
std::unique_ptr<mlir::Pass> createLowerGemmToRuntimePass();

void registerLadderPasses() {
  // placeholder for future pass registrations
}

} // namespace ladder
