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
    inherit = Qt5Gui.class,
    target = "org.bytedeco.qt.Qt5Widgets",
    global = "org.bytedeco.qt.global.Qt5Widgets",
    helper = "org.bytedeco.qt.helper.Qt5Widgets",
    value = {
        @Platform(
            include = {
                "qboxlayout.h",
                "QAbstractButton",
                "QAbstractScrollArea",
                "QAbstractSpinBox",
                "QAction",
                "QApplication",
                "QCheckBox",
                "QComboBox",
                "QDialog",
                "QFrame",
                "QGridLayout",
                "QGroupBox",
                "QLabel",
                "QLayout",
                "QLineEdit",
                "QMenu",
                "QMessageBox",
                "QPushButton",
                "QSizePolicy",
                "QSpinBox",
                "QStyle",
                "QSystemTrayIcon",
                "QTextEdit",
                "QToolButton",
                "QWidget"
            },
            link = "Qt5Widgets@.5"
        )
    }
)
public class Qt5Widgets extends QtInfoMapper {

  @Override
  public void map(InfoMap infoMap) {
    super.map(infoMap);
    infoMap
        .put(new Info("Q_WIDGETS_EXPORT").cppTypes().annotations())

        // Java methods
        .put(new Info("QAbstractSpinBox::event")
            .javaText("@Virtual protected native @Cast(\"bool\") boolean event(QEvent event);"))
        .put(new Info("QToolButton::sizeHint").javaText(
            "@Virtual public native @ByVal @Const({false, false, true}) QSize sizeHint();"))

        .put(new Info("QMenu::setAsDockMenu").annotations("@Platform(\"macosx\")"))

        // Line patterns
        .put(new Info("qabstractbutton.h").linePatterns(matchClass("QAbstractButton"), matchEnd()))
        .put(new Info("qabstractscrollarea.h").linePatterns(
            matchClass("QAbstractScrollArea"), matchEnd()))
        .put(new Info("qabstractspinbox.h").linePatterns(
            matchClass("QAbstractSpinBox"), matchEnd()))
        .put(new Info("qaction.h").linePatterns(matchClass("QAction"), matchEnd()))
        .put(new Info("qapplication.h").linePatterns(matchClass("QApplication"), matchEnd()))
        .put(new Info("qboxlayout.h").linePatterns(
            "class Q_WIDGETS_EXPORT .*", "QT_END_NAMESPACE"))
        .put(new Info("qcheckbox.h").linePatterns(matchClass("QCheckBox"), matchEnd()))
        .put(new Info("qcombobox.h").linePatterns(matchClass("QComboBox"), matchEnd()))
        .put(new Info("qdialog.h").linePatterns(matchClass("QDialog"), matchEnd()))
        .put(new Info("qframe.h").linePatterns(matchClass("QFrame"), matchEnd()))
        .put(new Info("qgridlayout.h").linePatterns(matchClass("QGridLayout"), matchEnd()))
        .put(new Info("qgroupbox.h").linePatterns(matchClass("QGroupBox"), matchEnd()))
        .put(new Info("qlabel.h").linePatterns(matchClass("QLabel"), matchEnd()))
        .put(new Info("qlayout.h").linePatterns(matchClass("QLayout"), matchEnd()))
        .put(new Info("qlineedit.h").linePatterns(
            matchClass("QLineEdit"), "#if QT_CONFIG\\(action\\)",
            "#endif", matchEnd()))
        .put(new Info("qmenu.h").linePatterns(matchClass("QMenu"), matchEnd()))
        .put(new Info("qmessagebox.h").linePatterns(matchClass("QMessageBox"), matchEnd()))
        .put(new Info("qpushbutton.h").linePatterns(matchClass("QPushButton"), matchEnd()))
        .put(new Info("qsizepolicy.h").linePatterns(matchClass("QSizePolicy"), matchEnd()))
        .put(new Info("qspinbox.h").linePatterns(matchClass("QSpinBox"), matchEnd()))
        .put(new Info("qstyle.h").linePatterns(matchClass("QStyle"), matchEnd()))
        .put(new Info("qsystemtrayicon.h").linePatterns(matchClass("QSystemTrayIcon"), matchEnd()))
        .put(new Info("qtextedit.h").linePatterns(matchClass("QTextEdit"), matchEnd()))
        .put(new Info("qtoolbutton.h").linePatterns(matchClass("QToolButton"), matchEnd()))
        .put(new Info("qwidget.h").linePatterns(matchClass("QWidget"), matchEnd()));
  }

  @Override
  protected String[] defineFalse() {
    return new String[]{
        // QSizePolicy
        "defined(Q_COMPILER_UNIFORM_INIT) && !defined(Q_QDOC)",

        // QWidget
        "QT_KEYPAD_NAVIGATION"
    };
  }

  @Override
  protected String[] enums() {
    return new String[]{
        "QAbstractSpinBox::StepType",
        "QSystemTrayIcon::ActivationReason",
        "QSystemTrayIcon::MessageIcon",
        "QTextEdit::AutoFormattingFlag"
    };
  }

  @Override
  protected Map<String, String> flags() {
    Map<String, String> flags = new HashMap<>();
    flags.put("QTextEdit::AutoFormatting", "QTextEdit::AutoFormattingFlag");
    return flags;
  }

  @Override
  protected String[] intEnums() {
    return new String[]{
        "QMessageBox::StandardButton",
        "QTextDocument::MarkdownFeatures"
    };
  }

  @Override
  protected String[] intFlags() {
    return new String[]{
        "QMessageBox::StandardButtons"
    };
  }

  @Override
  protected Map<String, String> macros() {
    Map<String, String> macros = new HashMap<>();
    macros.put("QT_SIZEPOLICY_CONSTEXPR", "");
    return macros;
  }

  @Override
  protected String[] skip() {
    return new String[]{
        // Classes
        "QAbstractItemDelegate",
        "QAbstractItemView",
        "QActionGroup",
        "QButtonGroup",
        "QCompleter",
        "QDesktopWidget",
        "QGraphicsEffect",
        "QGraphicsProxyWidget",
        "QPlatformMenu",
        "QScrollBar",
        "QSpacerItem",
        "QStyleHintReturn",
        "QStyleOption",
        "QWidgetList",

        // Members
        "QAbstractSpinBox::StepEnabled",
        "QWidget::changeEvent",
        "QWidget::enterEvent",
        "QWidget::focusNextPrevChild",
        "QWidget::hasHeightForWidth",
        "QWidget::heightForWidth",
        "QWidget::inputMethodQuery",
        "QWidget::leaveEvent",
        "QWidget::minimumSizeHint",
        "QWidget::nativeEvent",
        "QWidget::paintEvent",
        "QWidget::setupUi",
        "QWidget::setVisible",
        "QWidget::sizeHint",

        // Types
        "NSMenu"
    };
  }

  @Override
  protected String[] virtual() {
    return new String[]{
        "QAbstractButton",
        "QAbstractScrollArea",
        "QAbstractSpinBox",
        "QAction",
        "QApplication",
        "QCheckBox",
        "QComboBox",
        "QDialog",
        "QFrame",
//        "QGridLayout",
        "QGroupBox",
        "QLabel",
//        "QLayout",
        "QLineEdit",
        "QMenu",
        "QMessageBox",
        "QPushButton",
        "QSizePolicy",
        "QSpinBox",
//        "QStyle",
        "QSystemTrayIcon",
        "QTextEdit",
        "QToolButton",
        "QWidget"
    };
  }
}
