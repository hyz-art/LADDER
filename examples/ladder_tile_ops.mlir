module {
  func.func @tile_ops(%arg0: tensor<128x128xf16>) -> tensor<64x64xf16> {
    %t0 = "ladder.tile.extract"(%arg0) {offsets=[0,0], sizes=[64,64], strides=[1,1], layout="blocked", layout_params=[16,16]} : (tensor<128x128xf16>) -> tensor<64x64xf16>
    %t1 = "ladder.tile.prefetch"(%t0) {level=1 : i64} : (tensor<64x64xf16>) -> tensor<64x64xf16>
    %t2 = "ladder.tile.copy"(%t1) {direction="global_to_local", layout="blocked", layout_params=[16,16]} : (tensor<64x64xf16>) -> tensor<64x64xf16>
    %t3 = "ladder.tile.map"(%t2) {map_fn="relu", axis_map=[0,1], layout="blocked", layout_params=[16,16]} : (tensor<64x64xf16>) -> tensor<64x64xf16>
    %t4 = "ladder.tile.pad"(%t3) {pad_low=[0,0], pad_high=[0,0], pad_inner=[0,0], pad_value=0.0 : f32, layout="blocked", layout_params=[16,16]} : (tensor<64x64xf16>) -> tensor<64x64xf16>
    %t5 = "ladder.tile.async_copy"(%t4) {direction="local_to_global", layout="blocked", layout_params=[16,16]} : (tensor<64x64xf16>) -> tensor<64x64xf16>
    return %t5 : tensor<64x64xf16>
  }
}
