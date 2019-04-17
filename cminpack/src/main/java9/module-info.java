module org.bytedeco.cminpack {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.openblas;
  exports org.bytedeco.cminpack.global;
  exports org.bytedeco.cminpack.presets;
}
