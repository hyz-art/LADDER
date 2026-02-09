#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path


def _map_precision(p: str) -> str:
    p = p.strip().lower()
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


def _find_onnx_mlir_output(base: str) -> Path:
    # onnx-mlir usually emits <base>_onnx.mlir or <base>.onnx.mlir
    candidates = glob.glob(base + "*onnx*.mlir")
    if candidates:
        return Path(candidates[0])
    # fallback: <base>.mlir
    if Path(base + ".mlir").exists():
        return Path(base + ".mlir")
    raise FileNotFoundError("Cannot locate onnx-mlir output MLIR file.")


def _extract_gemm_shapes(mlir_text: str):
    # Match onnx.MatMul or onnx.Gemm type signatures
    # Example: (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    pat = re.compile(r"onnx\.(MatMul|Gemm).*?: \(tensor<([0-9]+)x([0-9]+)x[^>]+>, tensor<([0-9]+)x([0-9]+)x[^>]+>\) ->")
    for line in mlir_text.splitlines():
        m = pat.search(line)
        if m:
            M = int(m.group(2))
            K = int(m.group(3))
            K2 = int(m.group(4))
            N = int(m.group(5))
            if K == K2:
                return M, N, K
    return None


def main():
    ap = argparse.ArgumentParser(description="Use onnx-mlir to emit ONNX-MLIR and convert to ladder IR.")
    ap.add_argument("--onnx", required=True, help="Input ONNX file")
    ap.add_argument("--onnx-mlir", default="/hyz/mlir/onnx-mlir/build/Release/bin/onnx-mlir", help="onnx-mlir binary path")
    ap.add_argument("--out-prefix", default="ladder_ir", help="Output prefix")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--accumulate", default="fp32")
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--output-scale", type=float, default=1.0)
    ap.add_argument("--tileM", type=int, default=64)
    ap.add_argument("--tileN", type=int, default=64)
    ap.add_argument("--tileK", type=int, default=16)
    ap.add_argument("--impl", type=str, default="ttile", choices=["loop", "ttile"])
    ap.add_argument("--fuse-relu", action="store_true")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    base = str(Path(args.out_prefix))
    cmd = [args.onnx_mlir, "--EmitONNXIR", "-o", base, str(onnx_path)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    mlir_path = _find_onnx_mlir_output(base)
    mlir_text = mlir_path.read_text()
    shapes = _extract_gemm_shapes(mlir_text)
    if not shapes:
        raise RuntimeError("Failed to find GEMM/MatMul shapes in emitted MLIR.")

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

    json_path = Path(args.out_prefix).with_suffix(".json")
    ladder_mlir_path = Path(args.out_prefix).with_suffix(".mlir")
    json_path.write_text(json.dumps(ir, indent=2, ensure_ascii=False))

    ladder_mlir = f"""// ladder IR (MLIR-like)\nmodule {{\n  %qa = ladder.quantize {{scale={ir['input_scale']}, precision={prec}}}\n  %qb = ladder.quantize {{scale={ir['input_scale']}, precision={prec}}}\n  %acc = ladder.gemm {{M={M}, N={N}, K={K}, tileM={ir['tileM']}, tileN={ir['tileN']}, tileK={ir['tileK']}, precision={prec}, accumulate={acc}, input_scale={ir['input_scale']}, output_scale={ir['output_scale']}, impl={ir['impl']}, fuse_relu={ir['fuse_relu']}}}\n  %out = ladder.dequantize {{scale={ir['output_scale']}, precision={prec}}}\n}}\n"""
    ladder_mlir_path.write_text(ladder_mlir)

    print(f"Wrote {json_path}")
    print(f"Wrote {ladder_mlir_path}")
    print(f"ONNX-MLIR file: {mlir_path}")


if __name__ == "__main__":
    main()