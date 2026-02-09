feat: 添加基于 tile 的 GEMM 路径、调度器与 autotuner 扩展

变更概述：

- 新增 `gemm_tiled_fused` 的实现选择器，支持两条实现路径：
  - `Loop`：保留原有基于循环的 tile-loop 实现，作为默认实现，保证兼容性。
  - `TTile`：新增基于 `tTile` 的 tile-level 实现（`gemm_tiled_fused_tiles`），使得后续可在 tile 级别施加 layout transform、fusion 和 IR 映射。
- 增加轻量 `Scheduler` 草案（`include/scheduler.h`、`src/core/scheduler.cpp`），提供 `recommendTiles`，用于根据设备特性推荐 tile 大小。
- 扩展 `AutoTuner`：
  - `tuneGEMMAdvanced`：支持在 tile 大小、低精度 `tType::Precision` 与 fusion（是否应用 ReLU）维度进行搜索。
  - `tuneGEMMWithScheduler`：基于 `Scheduler` 的推荐生成候选 tile 三元组并驱动 advanced tuner。
- Autotuner 结果持久化到 `autotune_results.csv`（追加模式），并新增 `tools/plot_results.py` 用于生成简单汇总图 `autotune_summary.png`。
- 添加演示程序 `tools/run_autotune.cpp` 并在 CMake 中添加 `run_autotune` 可执行文件，以便在容器或机器上快速运行小规模搜索得到 CSV。

实现理由：
- 为后续实现硬件感知的低精度优化（包括 layout transform、算子融合、shared/local buffer 分配等）打下结构性基础；
- 将 tile 级别的操作显式化，便于将来将这些操作映射到自定义方言或 MLIR pass；
- 提供调度器与 autotuner 的基本闭环，便于快速实验和参数收敛。

测试：
- 已在容器中运行 `./build/run_autotune` 的小搜索示例，生成 `autotune_results.csv`，并在控制台打印出最佳 tile（例如 `Best tile: (67,67,16)`）。

后续建议（PR 讨论里可展开）：
- 将 `tTile` 实现进一步支持 layout transform（blocked/packed layout）与 explicit copy/prefetch 接口；
- 将调度器扩展为设备插件（NVIDIA/AMD），并结合硬件参数（L1/L2 大小、SM 数量、warp/wavefront）形成 cost model；
- 将 `gemm_tiled_fused_tiles` 的内部运算降级为硬件原语（如 WMMA/cuBLAS 或 ROCm intrinsics）以获取真实性能提升。

请在 PR 页面审阅变更、运行示例并提出反馈。谢谢！
