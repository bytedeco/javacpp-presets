module org.bytedeco.pytorch {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.openblas;
  exports org.bytedeco.pytorch.global;
  exports org.bytedeco.pytorch.presets;
  exports org.bytedeco.pytorch.cuda;
  exports org.bytedeco.pytorch.gloo;
  exports org.bytedeco.pytorch.chrono;
  exports org.bytedeco.pytorch;
}
