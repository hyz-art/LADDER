#include "ladder/IR/LadderDialect.h"
#include "ladder/Transforms/Passes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                       llvm::cl::desc("<input mlir>"),
                                       llvm::cl::Required);
  llvm::cl::opt<std::string> outputFile("o", llvm::cl::desc("Output file"),
                                        llvm::cl::init("-"));
  llvm::cl::ParseCommandLineOptions(argc, argv, "ladder-opt\n");

  MLIRContext context;
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<ladder::LadderDialect>();

  llvm::SourceMgr sourceMgr;
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFile);
  if (!fileOrErr) {
    llvm::errs() << "Failed to open input: " << inputFile << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse MLIR file\n";
    return 1;
  }

  PassManager pm(&context);
  pm.addPass(ladder::createLowerONNXToLadderPass());
  pm.addPass(ladder::createLowerTileToRuntimePass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Pass pipeline failed\n";
    return 1;
  }

  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFile, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  module->print(output->os());
  output->keep();
  return 0;
}
