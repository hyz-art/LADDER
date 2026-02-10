//===- LadderOps.td - Ladder ops ---------------------------*- tablegen -*-===//

include "mlir/IR/OpBase.td"
include "ladder/IR/LadderDialect.td"

class Ladder_Op<string mnemonic, list<Trait> traits = []> : Op<Ladder_Dialect, mnemonic, traits>;

// elementwise add (for fusion)

def Ladder_AddOp : Ladder_Op<"add"> {
  let summary = "Elementwise add";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$out);
}

// scheduling primitives

def Ladder_SliceOp : Ladder_Op<"slice"> {
  let summary = "Slice tensor with offsets/sizes/strides";
  let arguments = (ins AnyTensor:$input,
                      ArrayAttr:$offsets, ArrayAttr:$sizes, ArrayAttr:$strides);
  let results = (outs AnyTensor:$output);
}

def Ladder_MapOp : Ladder_Op<"map"> {
  let summary = "Elementwise map with schedule annotation";
  let arguments = (ins AnyTensor:$input, StrAttr:$map_fn, ArrayAttr:$axis_map);
  let results = (outs AnyTensor:$output);
}

def Ladder_PadOp : Ladder_Op<"pad"> {
  let summary = "Pad tensor with low/high/inner padding";
  let arguments = (ins AnyTensor:$input,
                      ArrayAttr:$pad_low, ArrayAttr:$pad_high, ArrayAttr:$pad_inner,
                      F32Attr:$pad_value);
  let results = (outs AnyTensor:$output);
}

def Ladder_TransformOp : Ladder_Op<"transform"> {
  let summary = "Tensor layout transform (e.g. reshape/transpose)";
  let arguments = (ins AnyTensor:$input,
                      StrAttr:$transform_kind, ArrayAttr:$params);
  let results = (outs AnyTensor:$output);
}

// tile-level ops

def Ladder_TileExtractOp : Ladder_Op<"tile.extract"> {
  let summary = "Extract a tile with layout metadata";
  let arguments = (ins AnyTensor:$input,
                      ArrayAttr:$offsets, ArrayAttr:$sizes, ArrayAttr:$strides,
                      StrAttr:$layout, ArrayAttr:$layout_params);
  let results = (outs AnyTensor:$output);
}

def Ladder_TileMapOp : Ladder_Op<"tile.map"> {
  let summary = "Tile-level map with layout metadata";
  let arguments = (ins AnyTensor:$input,
                      StrAttr:$map_fn, ArrayAttr:$axis_map,
                      StrAttr:$layout, ArrayAttr:$layout_params);
  let results = (outs AnyTensor:$output);
}

def Ladder_TilePadOp : Ladder_Op<"tile.pad"> {
  let summary = "Tile-level pad with layout metadata";
  let arguments = (ins AnyTensor:$input,
                      ArrayAttr:$pad_low, ArrayAttr:$pad_high, ArrayAttr:$pad_inner,
                      F32Attr:$pad_value,
                      StrAttr:$layout, ArrayAttr:$layout_params);
  let results = (outs AnyTensor:$output);
}

def Ladder_TileCopyOp : Ladder_Op<"tile.copy"> {
  let summary = "Explicit copy between global/local tiles";
  let arguments = (ins AnyTensor:$input,
                      StrAttr:$direction, StrAttr:$layout, ArrayAttr:$layout_params);
  let results = (outs AnyTensor:$output);
}

def Ladder_TilePrefetchOp : Ladder_Op<"tile.prefetch"> {
  let summary = "Prefetch tile into cache/local storage";
  let arguments = (ins AnyTensor:$input, I64Attr:$level);
  let results = (outs AnyTensor:$output);
}

def Ladder_TileAsyncCopyOp : Ladder_Op<"tile.async_copy"> {
  let summary = "Asynchronous copy between memory spaces";
  let arguments = (ins AnyTensor:$input,
                      StrAttr:$direction, StrAttr:$layout, ArrayAttr:$layout_params);
  let results = (outs AnyTensor:$output);
}

// quantize

def Ladder_QuantizeOp : Ladder_Op<"quantize"> {
  let summary = "Quantize tensor with tType precision";
  let arguments = (ins AnyTensor:$input, F32Attr:$scale, I64Attr:$zero_point,
                      F32Attr:$calibration, StrAttr:$precision);
  let results = (outs AnyTensor:$output);
}

// dequantize

def Ladder_DequantizeOp : Ladder_Op<"dequantize"> {
  let summary = "Dequantize tensor with tType precision";
  let arguments = (ins AnyTensor:$input, F32Attr:$scale, I64Attr:$zero_point,
                      F32Attr:$calibration, StrAttr:$precision);
  let results = (outs AnyTensor:$output);
}

// gemm

def Ladder_GemmOp : Ladder_Op<"gemm"> {
  let summary = "Tile-level GEMM with tType precision/accumulation";
  let arguments = (ins AnyTensor:$A, AnyTensor:$B,
                      I64Attr:$M, I64Attr:$N, I64Attr:$K,
                      I64Attr:$tileM, I64Attr:$tileN, I64Attr:$tileK,
                      StrAttr:$precision_a, StrAttr:$precision_b, StrAttr:$precision_c,
                      StrAttr:$accumulate, StrAttr:$accumulate_partial,
                      I64Attr:$accumulate_partial_block,
                      F32Attr:$input_scale, F32Attr:$output_scale,
                      StrAttr:$impl, I64Attr:$fuse_relu);
  let results = (outs AnyTensor:$C);
}

// fused gemm + bias (+ relu)

def Ladder_GemmFusedOp : Ladder_Op<"gemm_fused"> {
  let summary = "Fused GEMM + Bias (+ReLU)";
  let arguments = (ins AnyTensor:$A, AnyTensor:$B, AnyTensor:$Bias,
                      I64Attr:$M, I64Attr:$N, I64Attr:$K,
                      I64Attr:$tileM, I64Attr:$tileN, I64Attr:$tileK,
                      StrAttr:$precision_a, StrAttr:$precision_b, StrAttr:$precision_c,
                      StrAttr:$accumulate, StrAttr:$accumulate_partial,
                      I64Attr:$accumulate_partial_block,
                      F32Attr:$input_scale, F32Attr:$output_scale,
                      StrAttr:$impl, I64Attr:$fuse_relu);
  let results = (outs AnyTensor:$C);
}
