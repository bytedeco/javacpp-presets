module org.bytedeco.qt {
  requires transitive org.bytedeco.javacpp;
  exports org.bytedeco.qt.global;
  exports org.bytedeco.qt.QtCore;
  exports org.bytedeco.qt.QtGui;
  exports org.bytedeco.qt.QtWidgets;
}
