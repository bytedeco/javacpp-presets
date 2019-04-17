module org.bytedeco.systems {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.systems.global;
  exports org.bytedeco.systems.presets;
  exports org.bytedeco.systems.linux;
  exports org.bytedeco.systems.macosx;
  exports org.bytedeco.systems.windows;
}
