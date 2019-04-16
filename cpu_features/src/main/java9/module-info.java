module org.bytedeco.cpu_features {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.cpu_features.global;
  exports org.bytedeco.cpu_features.presets to org.bytedeco.javacpp;
  exports org.bytedeco.cpu_features;
}
