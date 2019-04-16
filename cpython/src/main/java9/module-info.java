module org.bytedeco.cpython {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.cpython.global;
  exports org.bytedeco.cpython.presets to org.bytedeco.javacpp;
  exports org.bytedeco.cpython;
}
