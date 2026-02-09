# Tools

## 前端导出（ONNX → ladder IR）

生成 JSON + MLIR-like 描述（默认 GEMM）：

```bash
python3 tools/onnx_to_ladder_ir.py --out-prefix ladder_ir
```

指定 ONNX（若已安装 onnx）：

```bash
python3 tools/onnx_to_ladder_ir.py --onnx path/to/model.onnx --out-prefix ladder_ir
```

生成一个简单的 GEMM ONNX 模型：

```bash
python3 tools/make_simple_gemm_onnx.py --out simple_gemm.onnx
```

## 前端导出（ONNX → onnx-mlir → ladder IR）

使用 onnx-mlir 生成 ONNX-MLIR，并从中抽取 GEMM 形状后生成 ladder IR：

```bash
python3 tools/onnx_mlir_to_ladder_ir.py --onnx path/to/model.onnx --out-prefix ladder_ir
```

## 闭环运行（MLIR-like → GEMM）

构建：

```bash
cmake -S . -B build
cmake --build build -- -j$(nproc)
```

运行：

```bash
./build/run_ladder_ir ladder_ir.mlir
```

## 可视化

```bash
python3 tools/plot_results.py
```

生成 `autotune_summary.png`。
