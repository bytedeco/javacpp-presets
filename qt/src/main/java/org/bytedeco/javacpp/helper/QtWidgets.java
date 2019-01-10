package org.bytedeco.javacpp.helper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.QtCore.*;
import org.bytedeco.javacpp.QtWidgets.*;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Opaque;

public class QtWidgets extends org.bytedeco.javacpp.presets.QtWidgets {

  @Name("[=](const QAbstractButton *sender, const QObject *context, void (*functor)(int, bool), int target){QObject::connect(sender, &QAbstractButton::clicked, context, [functor,target](bool checked){functor(target, checked);});}")
  public native static void QAbstractButton_clicked(@Const QAbstractButton sender, @Const QObject context, ClickedCallback functor, int target);

  public static class ClickedCallback extends FunctionPointer {
    static { Loader.load(); }
    private static List<ClickedCallback> cbs = Collections.synchronizedList(new ArrayList<ClickedCallback>());
    private static AtomicInteger ids = new AtomicInteger();
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ClickedCallback(Pointer p) { super(p); }
    protected ClickedCallback() { allocate(); id = ids.getAndIncrement(); cbs.add(id, this); }
    protected native void allocate();
    public int id;
    public final void call(int target, @Cast("bool") final boolean checked) {
      final ClickedCallback cb = cbs.get(target);
      new Thread(new Runnable() {
        @Override
        public void run() {
          if (cb != null) cb.clicked(checked);
        }
      }).start();
    }
    public void clicked(boolean checked) {}
  }

  @Opaque
  public static class QLayoutItem extends Pointer {
    public QLayoutItem() { super((Pointer)null); }
    public QLayoutItem(Pointer p) { super(p); }
  }
}
