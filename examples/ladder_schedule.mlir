module {
  func.func @schedule(%arg0: tensor<128x128xf16>) -> tensor<128x128xf16> {
    %s = "ladder.slice"(%arg0) {offsets = [0, 0], sizes = [128, 64], strides = [1, 1]} : (tensor<128x128xf16>) -> tensor<128x64xf16>
    %p = "ladder.pad"(%s) {pad_low = [0, 0], pad_high = [0, 64], pad_inner = [0, 0], pad_value = 0.0 : f32} : (tensor<128x64xf16>) -> tensor<128x128xf16>
    %m = "ladder.map"(%p) {map_fn = "relu", axis_map = [0, 1]} : (tensor<128x128xf16>) -> tensor<128x128xf16>
    %t = "ladder.transform"(%m) {transform_kind = "transpose", params = [1, 0]} : (tensor<128x128xf16>) -> tensor<128x128xf16>
    return %t : tensor<128x128xf16>
  }
}
