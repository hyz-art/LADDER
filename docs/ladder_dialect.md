# Ladder 方言（草案）

> 说明：本文件给出最小可用的 `ladder` 方言语义草案，用于描述 `tType` 量化与 GEMM 的 tile-level 表达。
> 当前实现以“MLIR-like 文本 IR + 运行时原型”为主，后续将迁移到 MLIR ODS/TableGen。

## 类型语义（tType）

`tType` 用于描述低精度计算及其混合精度策略，核心参数：

- `precision`：计算精度（fp32/fp16/fp8/int8/int4）
- `accumulate`：累加精度（通常为 fp32）
- `input_scale`：输入量化比例
- `output_scale`：输出反量化比例
- `zero_point`：量化零点（int）
- `calibration`：校准尺度（float，用于量化误差修正）

> 说明：当前原型使用 `quantize_sim` 来近似模拟低精度效果，真实硬件加速需在后端实现。

## 核心操作（Ops）

### `ladder.quantize`

语义：将输入张量按 `precision` 与 `scale/zero_point/calibration` 做量化。

示例：

```
%qa = ladder.quantize {scale=1.0, zero_point=0, calibration=1.0, precision=fp16}
```

### `ladder.dequantize`

语义：将输出张量按 `precision` 与 `scale/zero_point/calibration` 做反量化。

示例：

```
%out = ladder.dequantize {scale=1.0, zero_point=0, calibration=1.0, precision=fp16}
```

### `ladder.gemm`

语义：tile-level GEMM，支持融合、量化与累加精度控制。

属性（最小集合）：

- `M,N,K`：矩阵维度
- `tileM,tileN,tileK`：分块大小
- `precision_a/precision_b/precision_c`：输入/输出精度
- `accumulate`：主累加精度（通常为 fp32）
- `accumulate_partial`：分块累加精度
- `accumulate_partial_block`：分块次数（建议 $\lceil K / tileK \rceil$）
- `input_scale` / `output_scale`
- `impl`：实现选择（`loop`/`ttile`）
- `fuse_relu`：是否融合 ReLU

示例：

```
%acc = ladder.gemm {M=128, N=128, K=64, tileM=64, tileN=64, tileK=16,
                    precision_a=fp16, precision_b=fp16, precision_c=fp16,
                    accumulate=fp32, accumulate_partial=fp32,
                    accumulate_partial_block=4,
                    input_scale=1.0, output_scale=1.0,
                    impl=ttile, fuse_relu=0}

## 张量调度原语（Scheduling Primitives）

### `ladder.slice`

语义：对张量做切片，`offsets/sizes/strides` 为各维参数。

示例：

```
%s = ladder.slice %x {offsets=[0,0], sizes=[64,64], strides=[1,1]}
```

### `ladder.map`

语义：为元素级 map/向量化/并行化做标注。

示例：

```
%m = ladder.map %x {map_fn="relu", axis_map=[0,1]}
```

### `ladder.pad`

语义：对张量进行填充。

示例：

```
%p = ladder.pad %x {pad_low=[0,0], pad_high=[0,64], pad_inner=[0,0], pad_value=0.0}
```

### `ladder.transform`

语义：布局/形状转换（例如 reshape/transpose）。

示例：

```
%t = ladder.transform %x {transform_kind="transpose", params=[1,0]}
```

## Tile-level Ops（tTile）

用于表达 tile-level 的数据搬运、布局与预取，面向后端优化。

### `ladder.tile.extract`

语义：从全局张量中提取 tile，并携带布局元数据。

```
%tile = ladder.tile.extract %x {offsets=[0,0], sizes=[64,64], strides=[1,1],
                                layout="blocked", layout_params=[16,16]}
```

### `ladder.tile.map`

语义：对 tile 进行 map 标注（向量化/并行化）。

```
%tm = ladder.tile.map %tile {map_fn="relu", axis_map=[0,1],
                             layout="blocked", layout_params=[16,16]}
```

### `ladder.tile.pad`

语义：对 tile 进行填充（保持布局元数据）。

```
%tp = ladder.tile.pad %tile {pad_low=[0,0], pad_high=[0,16], pad_inner=[0,0],
                             pad_value=0.0, layout="blocked", layout_params=[16,16]}
```

### `ladder.tile.copy`

语义：显式拷贝（global ↔ local）。

```
%local = ladder.tile.copy %tile {direction="global_to_local", layout="blocked", layout_params=[16,16]}
```

### `ladder.tile.prefetch`

语义：预取（level 表示 cache 层级）。

```
%pf = ladder.tile.prefetch %tile {level=1}
```

### `ladder.tile.async_copy`

语义：异步拷贝（global ↔ local）。

```
%ac = ladder.tile.async_copy %tile {direction="global_to_local", layout="blocked", layout_params=[16,16]}
```
```

## Pass 规划（下一步）

1. **ONNX → ladder**：将 `onnx.MatMul/Gemm` 映射为 `ladder.gemm`。
2. **量化下沉**：自动插入 `ladder.quantize/dequantize`，并配置 `tType` 精度。
3. **Tile/布局变换**：引入 `tile.extract/pad/map` 语义并实现 lowering。
4. **融合**：在 tile 级别融合 `Gemm + Bias + Relu` 等链路。
5. **后端映射**：将 `ladder.gemm` 降级到 NVIDIA/AMD 后端实现。
