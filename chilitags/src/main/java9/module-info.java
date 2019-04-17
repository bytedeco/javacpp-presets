module org.bytedeco.chilitags {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.opencv;
  exports org.bytedeco.chilitags.global;
  exports org.bytedeco.chilitags.presets;
  exports org.bytedeco.chilitags;
}
