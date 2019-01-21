package org.bytedeco.javacpp.helper;

import java.io.File;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.QtCore.QString;

public class QtCore extends org.bytedeco.javacpp.presets.QtCore {

  static {
    // Load macOS framework
    File framework = new File("/usr/local/Cellar/qt/5.12.0/lib/QtCore.framework/QtCore");
    if (framework.exists()) {
      System.load(framework.getAbsolutePath());
    }

    // Load preset
    Loader.load(org.bytedeco.javacpp.QtCore.class);

    // Set main thread
    QtCore_verifyMainThread();
  }

  /**
   * Sets the main Qt thread to the current Java thread
   */
  public static native void QtCore_verifyMainThread();

  public abstract static class AbstractQString extends Pointer {

    protected AbstractQString(Pointer pointer) {
      super(pointer);
    }

    public QString add(QString s) {
      QString t = new QString(this);
      t.addPut(s);
      return t;
    }

    public abstract int compare(QString s);

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj instanceof QString) {
        return compare((QString) obj) == 0;
      }
      if (obj instanceof String) {
        return toString().equals(obj);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return toString().hashCode();
    }

    public abstract String toStdString();

    @Override
    public String toString() {
      return toStdString();
    }
  }
}
