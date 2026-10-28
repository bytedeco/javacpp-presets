module org.bytedeco.hdf5_plugins {
  requires transitive org.bytedeco.javacpp;
  requires transitive org.bytedeco.hdf5;
  exports org.bytedeco.hdf5_plugins.global;
  exports org.bytedeco.hdf5_plugins;
  exports org.bytedeco.hdf5_plugins.presets;
}
