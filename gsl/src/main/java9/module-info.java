module org.bytedeco.gsl {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.openblas;
  exports org.bytedeco.gsl.global;
  exports org.bytedeco.gsl.presets;
  exports org.bytedeco.gsl;
}
