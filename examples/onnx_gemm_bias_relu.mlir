module {
  func.func @main_graph(%arg0: tensor<128x64xf32>, %arg1: tensor<64x128xf32>, %bias: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    %1 = "onnx.Add"(%0, %bias) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = "onnx.Relu"(%1) : (tensor<128x128xf32>) -> tensor<128x128xf32>
    return %2 : tensor<128x128xf32>
  }
}
