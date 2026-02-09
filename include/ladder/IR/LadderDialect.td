//===- LadderDialect.td - Ladder dialect ------------------*- tablegen -*-===//

include "mlir/IR/DialectBase.td"

// Dialect definition

def Ladder_Dialect : Dialect {
  let name = "ladder";
  let cppNamespace = "::ladder";
  let summary = "Ladder dialect";
  let description = [{
    Ladder dialect models tile-level ops with tType precision semantics.
  }];
}
