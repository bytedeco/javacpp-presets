module org.bytedeco.onnx {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.onnx.global;
  exports org.bytedeco.onnx.presets to org.bytedeco.javacpp;
  exports org.bytedeco.onnx;
}
