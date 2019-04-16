module org.bytedeco.videoinput {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.videoinput.global;
  exports org.bytedeco.videoinput.presets to org.bytedeco.javacpp;
  exports org.bytedeco.videoinput;
}
