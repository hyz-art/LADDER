#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

try:
    import onnx  # optional
    from onnx import shape_inference
except Exception:
    onnx = None
    shape_inference = None


def _lower(s: str) -> str:
    return s.strip().lower()


def _map_precision(p: str) -> str:
    p = _lower(p)
    if p in {"fp32", "f32"}:
        return "fp32"
    if p in {"fp16", "f16"}:
        return "fp16"
    if p in {"fp8", "f8"}:
        return "fp8"
    if p in {"int8", "i8"}:
        return "int8"
    if p in {"int4", "i4"}:
        return "int4"
    return "fp32"


def _find_gemm_shapes(model):
    # Try to infer shapes and find a GEMM/MatMul node
    if shape_inference is not None:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass
    # Build a map of value_info shapes
    def _get_shape(vi):
        if not vi.type.tensor_type.shape.dim:
            return None
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                return None
        return dims

    shape_map = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        s = _get_shape(vi)
        if s:
            shape_map[vi.name] = s

    for node in model.graph.node:
        if node.op_type in {"Gemm", "MatMul"}:
            a = shape_map.get(node.input[0])
            b = shape_map.get(node.input[1])
            if a and b and len(a) == 2 and len(b) == 2:
                M, K = a
                K2, N = b
                if K2 == K:
                    return M, N, K
    return None


def main():
    ap = argparse.ArgumentParser(description="Convert a simple ONNX GEMM to ladder IR (JSON + MLIR-like).")
    ap.add_argument("--onnx", type=str, help="Path to ONNX model (optional)")
    ap.add_argument("--out-prefix", type=str, default="ladder_ir", help="Output file prefix")
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--precision", type=str, default="fp16")
    ap.add_argument("--accumulate", type=str, default="fp32")
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--output-scale", type=float, default=1.0)
    ap.add_argument("--tileM", type=int, default=64)
    ap.add_argument("--tileN", type=int, default=64)
    ap.add_argument("--tileK", type=int, default=16)
    ap.add_argument("--impl", type=str, default="ttile", choices=["loop", "ttile"])
    ap.add_argument("--fuse-relu", action="store_true")
    args = ap.parse_args()

    M, N, K = args.M, args.N, args.K
    if args.onnx:
        if onnx is None:
            print("onnx package not installed; using provided M/N/K.")
        else:
            model = onnx.load(args.onnx)
            shapes = _find_gemm_shapes(model)
            if shapes:
                M, N, K = shapes

    prec = _map_precision(args.precision)
    acc = _map_precision(args.accumulate)

    ir = {
        "op": "gemm",
        "M": M,
        "N": N,
        "K": K,
        "tileM": args.tileM,
        "tileN": args.tileN,
        "tileK": args.tileK,
        "precision": prec,
        "accumulate": acc,
        "input_scale": args.input_scale,
        "output_scale": args.output_scale,
        "impl": args.impl,
        "fuse_relu": 1 if args.fuse_relu else 0,
    }

    out_prefix = Path(args.out_prefix)
    json_path = out_prefix.with_suffix(".json")
    mlir_path = out_prefix.with_suffix(".mlir")

    json_path.write_text(json.dumps(ir, indent=2, ensure_ascii=False))

    mlir = f"""// ladder IR (MLIR-like)\nmodule {{\n  // quantize inputs\n  %qa = ladder.quantize {{scale={ir['input_scale']}, precision={prec}}}\n  %qb = ladder.quantize {{scale={ir['input_scale']}, precision={prec}}}\n  // GEMM\n  %acc = ladder.gemm {{M={M}, N={N}, K={K}, tileM={ir['tileM']}, tileN={ir['tileN']}, tileK={ir['tileK']}, precision={prec}, accumulate={acc}, input_scale={ir['input_scale']}, output_scale={ir['output_scale']}, impl={ir['impl']}, fuse_relu={ir['fuse_relu']}}}\n  // dequantize outputs\n  %out = ladder.dequantize {{scale={ir['output_scale']}, precision={prec}}}\n}}\n"""
    mlir_path.write_text(mlir)

    print(f"Wrote {json_path}")
    print(f"Wrote {mlir_path}")


if __name__ == "__main__":
    main()
