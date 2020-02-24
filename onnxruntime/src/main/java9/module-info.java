module org.bytedeco.onnxruntime {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.dnnl;
  exports org.bytedeco.onnxruntime.global;
  exports org.bytedeco.onnxruntime.presets;
  exports org.bytedeco.onnxruntime;
}
