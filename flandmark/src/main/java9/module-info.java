module org.bytedeco.flandmark {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.opencv;
  exports org.bytedeco.flandmark.global;
  exports org.bytedeco.flandmark.presets;
  exports org.bytedeco.flandmark;
}
