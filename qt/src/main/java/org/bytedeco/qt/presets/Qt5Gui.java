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

package org.bytedeco.qt.presets;

import java.util.HashMap;
import java.util.Map;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;

@Properties(
    inherit = Qt5Core.class,
    target = "org.bytedeco.qt.Qt5Gui",
    helper = "org.bytedeco.qt.helper.Qt5Gui",
    global = "org.bytedeco.qt.global.Qt5Gui",
    value = {
        @Platform(
            include = {
                "qevent.h",
                "qtextdocument.h",
                "QFont",
                "QIcon",
                "QGuiApplication"
            },
            link = "Qt5Gui@.5",
            preload = {"Qt5DBus@.5", "Qt5Gui@.5", "Qt5XcbQpa@.5", "Qt5Widgets@.5", "Qt5PrintSupport@.5",
                       "qmacstyle", "qcocoa", "cocoaprintersupport", "qgtk3", "qxdgdesktopportal",
                       "qxcb", "qlinuxfb", "qminimalegl", "qminimal", "qoffscreen", // "qtuiotouchplugin",
                       "composeplatforminputcontextplugin", "ibusplatforminputcontextplugin",
                       "qxcb-egl-integration", "qxcb-glx-integration", "qgif", "qico", "qjpeg",
                       "qdirect2d", "qwindows", "qwindowsvistastyle", "windowsprintersupport", // "qgenericbearer",
                       "qevdevkeyboardplugin", "qevdevmouseplugin", "qevdevtabletplugin", "qevdevtouchplugin"}
        )
    }
)
public class Qt5Gui extends QtInfoMapper {

  @Override
  public void map(InfoMap infoMap) {
    super.map(infoMap);
    infoMap
        .put(new Info("Q_GUI_EXPORT").cppTypes().annotations())

        // Line patterns
        .put(new Info("qevent.h").linePatterns(
            matchClass("QCloseEvent"), matchEnd()))
        .put(new Info("qfont.h").linePatterns(matchClass("QFont"), matchEnd()))
        .put(new Info("qicon.h").linePatterns(matchClass("QIcon"), matchEnd()))
        .put(new Info("qguiapplication.h").linePatterns(matchClass("QGuiApplication"), matchEnd()))

        // Members
        .put(new Info("QFont::toString").javaNames("toQString"));
  }

  @Override
  protected String[] enums() {
    return new String[]{
        "QPalette:ColorRole"
    };
  }

  @Override
  protected Map<String, String> macros() {
    Map<String, String> macros = new HashMap<>();

    // Types
    macros.put("WId", "size_t");

    return macros;
  }

  @Override
  protected String[] skip() {
    return new String[]{
        // Classes
        "QActionEvent",
        "QBackingStore",
        "QBitmap",
        "QClipboard",
        "QColor",
        "QContextMenuEvent",
        "QCursor",
        "QDragEnterEvent",
        "QDragLeaveEvent",
        "QDragMoveEvent",
        "QDropEvent",
        "QFocusEvent",
        "QFontInfo",
        "QFontMetrics",
        "QHideEvent",
        "QIconEngine",
        "QInputMethod",
        "QInputMethodEvent",
        "QKeyEvent",
        "QKeySequence",
        "QMouseEvent",
        "QMoveEvent",
        "QMovie",
        "QPagedPaintDevice",
        "QPaintEngine",
        "QPainter",
        "QPalette",
        "QPicture",
        "QPixmap",
        "QPlatformNativeInterface",
        "QRegion",
        "QResizeEvent",
        "QScreen",
        "QShowEvent",
        "QStyleHints",
        "QTabletEvent",
        "QTextCharFormat",
        "QTextDocument",
        "QTextCursor",
        "QValidator",
        "QWheelEvent",
        "QWindow",
        "QWindowList",

        // Enums
        "QKeySequence::StandardKey",
        "QPalette::ColorRole",
        "QTextCursor::MoveMode",
        "QTextCursor::MoveOperation",
        "QTextDocument::FindFlags",
        "QTextOption::WrapMode",
        "QValidator::State",

        // Members
        "QFont::resolve"
    };
  }

  @Override
  protected String[] virtual() {
    return new String[]{
        "QFont",
        "QIcon",
        "QGuiApplication"
    };
  }
}
