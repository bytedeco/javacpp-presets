module org.bytedeco.caffe {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.hdf5;
  requires transitive org.bytedeco.opencv;
  requires transitive org.bytedeco.openblas;
  exports org.bytedeco.caffe.global;
  exports org.bytedeco.caffe.presets;
  exports org.bytedeco.caffe;
}
