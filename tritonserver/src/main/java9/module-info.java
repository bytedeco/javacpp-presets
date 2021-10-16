module org.bytedeco.tritonserver {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.cuda;
  requires transitive org.bytedeco.tensorrt;
  exports org.bytedeco.tritonserver.global;
  exports org.bytedeco.tritonserver.presets;
  exports org.bytedeco.tritonserver;
}
