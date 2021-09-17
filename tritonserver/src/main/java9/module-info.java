module org.bytedeco.tritonserver {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.cuda;
  requires transitive org.bytedeco.tensorrt;
  exports org.bytedeco.tensorrt.global;
  exports org.bytedeco.tensorrt.presets;
  exports org.bytedeco.tensorrt.tritonserver;
}
