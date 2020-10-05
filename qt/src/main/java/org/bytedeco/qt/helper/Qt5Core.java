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
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.qt.Qt5Core.QString;

public class Qt5Core extends org.bytedeco.qt.presets.Qt5Core {

  static {
//    // Load macOS framework
//    File framework = new File("/usr/local/Cellar/qt/5.15.1/lib/QtCore.framework/QtCore");
//    if (framework.exists()) {
//      System.load(framework.getAbsolutePath());
//    }

    // Load preset
    Loader.load(org.bytedeco.qt.presets.Qt5Core.class);

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
