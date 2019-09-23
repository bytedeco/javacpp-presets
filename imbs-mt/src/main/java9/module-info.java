module org.bytedeco.imbs {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.opencv;
  exports org.bytedeco.imbs.global;
  exports org.bytedeco.imbs.presets;
  exports org.bytedeco.imbs;
}
