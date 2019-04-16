module org.bytedeco.libfreenect {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.libfreenect.global;
  exports org.bytedeco.libfreenect.presets to org.bytedeco.javacpp;
  exports org.bytedeco.libfreenect;
}
