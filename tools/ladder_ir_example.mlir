// ladder IR (MLIR-like)
module {
  %qa = ladder.quantize {scale=1.0, precision=fp16}
  %qb = ladder.quantize {scale=1.0, precision=fp16}
  %acc = ladder.gemm {M=128, N=128, K=64, tileM=64, tileN=64, tileK=16, precision=fp16, accumulate=fp32, input_scale=1.0, output_scale=1.0, impl=ttile, fuse_relu=0}
  %out = ladder.dequantize {scale=1.0, precision=fp16}
}
