module org.bytedeco.openpose {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.caffe;
  requires transitive org.bytedeco.hdf5;
  requires transitive org.bytedeco.opencv;
  requires transitive org.bytedeco.openblas;
  exports org.bytedeco.openpose.global;
  exports org.bytedeco.openpose.presets;
  exports org.bytedeco.openpose;
}
