module org.bytedeco.ngraph {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.openblas;
  exports org.bytedeco.ngraph.global;
  exports org.bytedeco.ngraph.presets;
  exports org.bytedeco.ngraph;
}
