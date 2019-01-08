package org.bytedeco.javacpp.helper;

import org.bytedeco.javacpp.Pointer;

public class QtCore extends org.bytedeco.javacpp.presets.QtCore {

  public abstract static class AbstractQString extends Pointer {

    protected AbstractQString(Pointer pointer) {
      super(pointer);
    }

    public abstract String toStdString();

    @Override
    public String toString() {
      return toStdString();
    }
  }
}
