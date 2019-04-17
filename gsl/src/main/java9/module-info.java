module org.bytedeco.gsl {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.openblas;
  exports org.bytedeco.gsl.global;
  exports org.bytedeco.gls.presets;
  exports org.bytedeco.gsl;
}
