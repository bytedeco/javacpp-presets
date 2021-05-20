module org.bytedeco.libffi {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.libffi.global;
  exports org.bytedeco.libffi.presets;
  exports org.bytedeco.libffi;
}
