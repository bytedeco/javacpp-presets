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
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Opaque;

public class Qt5Gui extends org.bytedeco.qt.presets.Qt5Gui {

//  static {
//    File framework = new File("/usr/local/Cellar/qt/5.15.1/lib/QtGui.framework/QtGui");
//    if (framework.exists()) {
//      System.load(framework.getAbsolutePath());
//    }
//  }

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
