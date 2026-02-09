//===- LadderOps.td - Ladder ops ---------------------------*- tablegen -*-===//

include "mlir/IR/OpBase.td"
include "ladder/IR/LadderDialect.td"

class Ladder_Op<string mnemonic, list<Trait> traits = []> : Op<Ladder_Dialect, mnemonic, traits>;

// quantize

def Ladder_QuantizeOp : Ladder_Op<"quantize"> {
  let summary = "Quantize tensor with tType precision";
  let arguments = (ins AnyTensor:$input, F32Attr:$scale, StrAttr:$precision);
  let results = (outs AnyTensor:$output);
}

// dequantize

def Ladder_DequantizeOp : Ladder_Op<"dequantize"> {
  let summary = "Dequantize tensor with tType precision";
  let arguments = (ins AnyTensor:$input, F32Attr:$scale, StrAttr:$precision);
  let results = (outs AnyTensor:$output);
}

// gemm

def Ladder_GemmOp : Ladder_Op<"gemm"> {
  let summary = "Tile-level GEMM with tType precision/accumulation";
  let arguments = (ins AnyTensor:$A, AnyTensor:$B,
                      I64Attr:$M, I64Attr:$N, I64Attr:$K,
                      I64Attr:$tileM, I64Attr:$tileN, I64Attr:$tileK,
                      StrAttr:$precision, StrAttr:$accumulate,
                      F32Attr:$input_scale, F32Attr:$output_scale,
                      StrAttr:$impl, I64Attr:$fuse_relu);
  let results = (outs AnyTensor:$C);
}
