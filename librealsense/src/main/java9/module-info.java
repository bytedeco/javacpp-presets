module org.bytedeco.librealsense {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.librealsense.global;
  exports org.bytedeco.librealsense.presets to org.bytedeco.javacpp;
  exports org.bytedeco.librealsense;
}
