package org.bytedeco.javacpp.helper;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Opaque;

public class QtGui extends org.bytedeco.javacpp.presets.QtGui {

  @Opaque
  public static class QPaintDevice extends Pointer {
    public QPaintDevice() { super((Pointer)null); }
    public QPaintDevice(Pointer p) { super(p); }
  }
}
