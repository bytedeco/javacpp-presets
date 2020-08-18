module org.bytedeco.qt {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.qt.global;
  exports org.bytedeco.qt.helper;
  exports org.bytedeco.qt.presets;
  exports org.bytedeco.qt.Qt5Core;
  exports org.bytedeco.qt.Qt5Gui;
  exports org.bytedeco.qt.Qt5Widgets;
}
