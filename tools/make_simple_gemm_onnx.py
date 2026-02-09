#!/usr/bin/env python3
import argparse
import onnx
from onnx import helper, TensorProto


def main():
    ap = argparse.ArgumentParser(description="Create a simple GEMM/MatMul ONNX model.")
    ap.add_argument("--out", default="simple_gemm.onnx")
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--K", type=int, default=64)
    args = ap.parse_args()

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [args.M, args.K])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [args.K, args.N])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [args.M, args.N])

    node = helper.make_node("MatMul", ["A", "B"], ["Y"], name="MatMul")

    graph = helper.make_graph([node], "simple_gemm", [A, B], [Y])
    model = helper.make_model(graph, producer_name="ladder-demo")
    onnx.save(model, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()