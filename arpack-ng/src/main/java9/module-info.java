module org.bytedeco.arpackng {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.openblas;
  exports org.bytedeco.arpackng.global;
  exports org.bytedeco.arpackng.presets;
}
