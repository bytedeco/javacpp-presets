package org.bytedeco.javacpp.helper;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Opaque;

public class QtWidgets extends org.bytedeco.javacpp.presets.QtWidgets {

  @Opaque
  public static class QLayoutItem extends Pointer {
    public QLayoutItem() { super((Pointer)null); }
    public QLayoutItem(Pointer p) { super(p); }
  }
}
