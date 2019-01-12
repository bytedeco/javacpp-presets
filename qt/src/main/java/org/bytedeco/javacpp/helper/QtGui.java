package org.bytedeco.javacpp.helper;

import java.io.File;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Opaque;

public class QtGui extends org.bytedeco.javacpp.presets.QtGui {

  static {
    File framework = new File("/usr/local/Cellar/qt/5.12.0/lib/QtGui.framework/QtGui");
    if (framework.exists()) {
      System.load(framework.getAbsolutePath());
    }
  }

  @Opaque
  public static class QPaintDevice extends Pointer {
    public QPaintDevice() { super((Pointer)null); }
    public QPaintDevice(Pointer p) { super(p); }
  }

  @Opaque
  public static class QPaintEvent extends Pointer {
    public QPaintEvent() { super((Pointer)null); }
    public QPaintEvent(Pointer p) { super(p); }
  }
}
