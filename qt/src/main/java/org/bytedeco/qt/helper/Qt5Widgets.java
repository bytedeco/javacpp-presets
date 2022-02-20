/*
 * Copyright (C) 2019 Greg Hart, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.qt.helper;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Allocator;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Opaque;
import org.bytedeco.qt.Qt5Core.*;
import org.bytedeco.qt.Qt5Widgets.*;

public class Qt5Widgets extends org.bytedeco.qt.presets.Qt5Widgets {

//  static {
//    File framework = new File("/usr/local/Cellar/qt/5.15.1/lib/QtWidgets.framework/QtWidgets");
//    if (framework.exists()) {
//      System.load(framework.getAbsolutePath());
//    }
//  }

  @Name("[=](const QAbstractButton *sender, const QObject *context, void (*functor)(int, bool), int target){QObject::connect(sender, &QAbstractButton::clicked, context, [functor,target](bool checked){functor(target, checked);});}")
  public native static void QAbstractButton_clicked(@Const QAbstractButton sender, @Const QObject context, ClickedCallback functor, int target);

  @Allocator(max = 100)
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

  @Name("[=](const QAbstractButton *sender, const QObject *context, void (*functor)(int, bool), int target){QObject::connect(sender, &QAbstractButton::toggled, context, [functor,target](bool checked){functor(target, checked);});}")
  public native static void QAbstractButton_toggled(@Const QAbstractButton sender, @Const QObject context, ToggledCallback functor, int target);

  @Allocator(max = 100)
  public static class ToggledCallback extends FunctionPointer {
    static { Loader.load(); }
    private static List<ToggledCallback> cbs = Collections.synchronizedList(new ArrayList<ToggledCallback>());
    private static AtomicInteger ids = new AtomicInteger();
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ToggledCallback(Pointer p) { super(p); }
    protected ToggledCallback() { allocate(); id = ids.getAndIncrement(); cbs.add(id, this); }
    protected native void allocate();
    public int id;
    public final void call(int target, @Cast("bool") final boolean checked) {
      final ToggledCallback cb = cbs.get(target);
      cb.toggled(checked);
    }
    public void toggled(boolean checked) {}
  }

  @Name("[=](const QAction *sender, const QObject *context, void (*functor)(int, bool), int target){QObject::connect(sender, &QAction::triggered, context, [functor,target](bool checked){functor(target, checked);});}")
  public native static void QAction_triggered(@Const QAction sender, @Const QObject context, TriggeredCallback functor, int target);

  @Allocator(max = 100)
  public static class TriggeredCallback extends FunctionPointer {
    static { Loader.load(); }
    private static List<TriggeredCallback> cbs = Collections.synchronizedList(new ArrayList<TriggeredCallback>());
    private static AtomicInteger ids = new AtomicInteger();
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TriggeredCallback(Pointer p) { super(p); }
    protected TriggeredCallback() { allocate(); id = ids.getAndIncrement(); cbs.add(id, this); }
    protected native void allocate();
    public int id;
    public final void call(int target, @Cast("bool") final boolean checked) {
      final TriggeredCallback cb = cbs.get(target);
      new Thread(new Runnable() {
        @Override
        public void run() {
          if (cb != null) cb.triggered(checked);
        }
      }).start();
    }
    public void triggered(boolean checked) {}
  }

  @Name("[=](const QComboBox *sender, const QObject *context, void (*functor)(int, int), int target){QObject::connect(sender, QOverload<int>::of(&QComboBox::currentIndexChanged), context, [functor,target](int index){functor(target, index);});}")
  public native static void QComboBox_currentIndexChanged(@Const QComboBox sender, @Const QObject context, CurrentIndexChangedCallback1 functor, int target);

  public static class CurrentIndexChangedCallback1 extends FunctionPointer {
    static { Loader.load(); }
    private static List<CurrentIndexChangedCallback1> cbs = Collections.synchronizedList(new ArrayList<CurrentIndexChangedCallback1>());
    private static AtomicInteger ids = new AtomicInteger();
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CurrentIndexChangedCallback1(Pointer p) { super(p); }
    protected CurrentIndexChangedCallback1() { allocate(); id = ids.getAndIncrement(); cbs.add(id, this); }
    protected native void allocate();
    public int id;
    public final void call(int target, final int index) {
      final CurrentIndexChangedCallback1 cb = cbs.get(target);
      new Thread(new Runnable() {
        @Override
        public void run() {
          if (cb != null) cb.currentIndexChanged(index);
        }
      }).start();
    }
    public void currentIndexChanged(int index) {}
  }

  @Name("[=](const QSystemTrayIcon *sender, const QObject *context, void (*functor)(int, QSystemTrayIcon::ActivationReason), int target){QObject::connect(sender, &QSystemTrayIcon::activated, context, [functor,target](QSystemTrayIcon::ActivationReason reason){functor(target, reason);});}")
  public native static void QSystemTrayIcon_activated(@Const QSystemTrayIcon sender, @Const QObject context, ActivatedCallback functor, int target);

  @Allocator(max = 100)
  public static class ActivatedCallback extends FunctionPointer {
    static { Loader.load(); }
    private static List<ActivatedCallback> cbs = Collections.synchronizedList(new ArrayList<ActivatedCallback>());
    private static AtomicInteger ids = new AtomicInteger();
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ActivatedCallback(Pointer p) { super(p); }
    protected ActivatedCallback() { allocate(); id = ids.getAndIncrement(); cbs.add(id, this); }
    protected native void allocate();
    public int id;
    public final void call(int target, final QSystemTrayIcon.ActivationReason reason) {
      final ActivatedCallback cb = cbs.get(target);
      new Thread(new Runnable() {
        @Override
        public void run() {
          if (cb != null) cb.activated(reason);
        }
      }).start();
    }
    public void activated(QSystemTrayIcon.ActivationReason reason) {}
  }

  @Name("[=](const QSystemTrayIcon *sender, const QObject *context, void (*functor)(int), int target){QObject::connect(sender, &QSystemTrayIcon::messageClicked, context, [functor,target](){functor(target);});}")
  public native static void QSystemTrayIcon_messageClicked(@Const QSystemTrayIcon sender, @Const QObject context, MessageClickedCallback functor, int target);

  @Allocator(max = 100)
  public static class MessageClickedCallback extends FunctionPointer {
    static { Loader.load(); }
    private static List<MessageClickedCallback> cbs = Collections.synchronizedList(new ArrayList<MessageClickedCallback>());
    private static AtomicInteger ids = new AtomicInteger();
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MessageClickedCallback(Pointer p) { super(p); }
    protected MessageClickedCallback() { allocate(); id = ids.getAndIncrement(); cbs.add(id, this); }
    protected native void allocate();
    public int id;
    public final void call(int target) {
      final MessageClickedCallback cb = cbs.get(target);
      new Thread(new Runnable() {
        @Override
        public void run() {
          if (cb != null) cb.messageClicked();
        }
      }).start();
    }
    public void messageClicked() {}
  }

  @Opaque
  public static class QLayoutItem extends Pointer {
    public QLayoutItem() { super((Pointer)null); }
    public QLayoutItem(Pointer p) { super(p); }
  }
}
