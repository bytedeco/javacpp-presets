package org.bytedeco.javacpp.presets;

import java.util.HashMap;
import java.util.Map;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;

@Properties(
    helper = "org.bytedeco.javacpp.helper.QtWidgets",
    inherit = QtGui.class,
    target = "org.bytedeco.javacpp.QtWidgets",
    value = {
        @Platform(
            include = {
                "QAbstractButton",
                "QApplication",
                "QGridLayout",
                "QLayout",
                "QLineEdit",
                "QSizePolicy",
                "QToolButton",
                "QWidget"
            },
            preload = "Qt5Widgets@5"
        )
    }
)
public class QtWidgets extends AbstractQtPreset {

  @Override
  public void map(InfoMap infoMap) {
    super.map(infoMap);
    infoMap
        .put(new Info("Q_WIDGETS_EXPORT").cppTypes().annotations())

        // Line patterns
        .put(new Info("qabstractbutton.h").linePatterns(matchClass("QAbstractButton"), matchEnd()))
        .put(new Info("qapplication.h").linePatterns(matchClass("QApplication"), matchEnd()))
        .put(new Info("qgridlayout.h").linePatterns(matchClass("QGridLayout"), matchEnd()))
        .put(new Info("qlayout.h").linePatterns(matchClass("QLayout"), matchEnd()))
        .put(new Info("qlineedit.h").linePatterns(
            matchClass("QLineEdit"), "#if QT_CONFIG\\(action\\)",
            "#endif", matchEnd()))
        .put(new Info("qsizepolicy.h").linePatterns(
            matchClass("QSizePolicy"), "public:",
            " *enum ControlType \\{", matchEnd()
        ))
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
  protected Map<String, String> macros() {
    Map<String, String> macros = new HashMap<>();
    macros.put("QT_SIZEPOLICY_CONSTEXPR", "");
    return macros;
  }

  @Override
  protected String[] skip() {
    return new String[]{
        // Classes
        "QAction",
        "QButtonGroup",
        "QCompleter",
        "QDesktopWidget",
        "QGraphicsEffect",
        "QGraphicsProxyWidget",
        "QMenu",
        "QStyle",
        "QWidgetList",

        // Members
        "QWidget::setupUi"
    };
  }
}
