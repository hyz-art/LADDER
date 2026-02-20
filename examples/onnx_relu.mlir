module {
  func.func @main_graph(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
