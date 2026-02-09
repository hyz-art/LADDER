# Ladder 方言（草案）

> 说明：本文件给出最小可用的 `ladder` 方言语义草案，用于描述 `tType` 量化与 GEMM 的 tile-level 表达。
> 当前实现以“MLIR-like 文本 IR + 运行时原型”为主，后续将迁移到 MLIR ODS/TableGen。

## 类型语义（tType）

`tType` 用于描述低精度计算及其混合精度策略，核心参数：

- `precision`：计算精度（fp32/fp16/fp8/int8/int4）
- `accumulate`：累加精度（通常为 fp32）
- `input_scale`：输入量化比例
- `output_scale`：输出反量化比例

> 说明：当前原型使用 `quantize_sim` 来近似模拟低精度效果，真实硬件加速需在后端实现。

## 核心操作（Ops）

### `ladder.quantize`

语义：将输入张量按 `precision` 与 `input_scale` 做量化。

示例：

```
%qa = ladder.quantize {scale=1.0, precision=fp16}
```

### `ladder.dequantize`

语义：将输出张量按 `precision` 与 `output_scale` 做反量化。

示例：

```
%out = ladder.dequantize {scale=1.0, precision=fp16}
```

### `ladder.gemm`

语义：tile-level GEMM，支持融合、量化与累加精度控制。

属性（最小集合）：

- `M,N,K`：矩阵维度
- `tileM,tileN,tileK`：分块大小
- `precision`：计算精度
- `accumulate`：累加精度（通常为 fp32）
- `input_scale` / `output_scale`
- `impl`：实现选择（`loop`/`ttile`）
- `fuse_relu`：是否融合 ReLU

示例：

```
%acc = ladder.gemm {M=128, N=128, K=64, tileM=64, tileN=64, tileK=16,
                    precision=fp16, accumulate=fp32,
                    input_scale=1.0, output_scale=1.0,
                    impl=ttile, fuse_relu=0}
```

## Pass 规划（下一步）

1. **ONNX → ladder**：将 `onnx.MatMul/Gemm` 映射为 `ladder.gemm`。
2. **量化下沉**：自动插入 `ladder.quantize/dequantize`，并配置 `tType` 精度。
3. **Tile/布局变换**：引入 `tile.extract/pad/map` 语义并实现 lowering。
4. **融合**：在 tile 级别融合 `Gemm + Bias + Relu` 等链路。
5. **后端映射**：将 `ladder.gemm` 降级到 NVIDIA/AMD 后端实现。
