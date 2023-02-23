module org.bytedeco.hdf5 {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.hdf5.global;
  exports org.bytedeco.hdf5.presets;
  exports org.bytedeco.hdf5;

  exports hdf.hdf5lib;
  exports hdf.hdf5lib.callbacks;
  exports hdf.hdf5lib.exceptions;
  exports hdf.hdf5lib.structs;
}
