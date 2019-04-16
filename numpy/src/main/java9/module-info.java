module org.bytedeco.numpy {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.numpy.global;
  exports org.bytedeco.numpy.presets to org.bytedeco.javacpp;
  exports org.bytedeco.numpy;
}
