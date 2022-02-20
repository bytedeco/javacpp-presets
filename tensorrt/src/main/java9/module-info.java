module org.bytedeco.tensorrt {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.cuda;
  exports org.bytedeco.tensorrt.global;
  exports org.bytedeco.tensorrt.presets;
  exports org.bytedeco.tensorrt.nvinfer;
  exports org.bytedeco.tensorrt.nvinfer_plugin;
  exports org.bytedeco.tensorrt.nvonnxparser;
  exports org.bytedeco.tensorrt.nvparsers;
}
