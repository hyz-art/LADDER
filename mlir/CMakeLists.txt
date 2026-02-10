include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/mlir)

# Ensure tablegen outputs go to generated include tree
set(LLVM_TABLEGEN_OUTPUT_DIR ${PROJECT_BINARY_DIR}/mlir)
set(MLIR_TABLEGEN_OUTPUT_DIR ${PROJECT_BINARY_DIR}/mlir)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/mlir)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/ladder/IR/LadderDialect.td)
mlir_tablegen(LadderDialect.h.inc -gen-dialect-decls -dialect=ladder)
mlir_tablegen(LadderDialect.cpp.inc -gen-dialect-defs -dialect=ladder)
add_public_tablegen_target(LadderDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/ladder/IR/LadderOps.td)
mlir_tablegen(LadderOps.h.inc -gen-op-decls)
mlir_tablegen(LadderOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(LadderOpsIncGen)

add_mlir_library(LadderDialect
  ${PROJECT_SOURCE_DIR}/lib/IR/LadderDialect.cpp
  ${PROJECT_SOURCE_DIR}/lib/IR/LadderOps.cpp
  DEPENDS
    LadderDialectIncGen
    LadderOpsIncGen
  LINK_LIBS PUBLIC
    MLIRIR
)

target_include_directories(LadderDialect PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}/mlir
)

add_mlir_library(LadderTransforms
  ${PROJECT_SOURCE_DIR}/lib/Transforms/LowerONNXToLadder.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/FuseLadderOps.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/LowerFusedGemmToRuntime.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/LowerTileToRuntime.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/Passes.cpp
  LINK_LIBS PUBLIC
    LadderDialect
    MLIRIR
    MLIRPass
)

target_include_directories(LadderTransforms PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}/mlir
)
