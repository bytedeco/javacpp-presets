package org.bytedeco.javacpp.presets;

import java.util.HashMap;
import java.util.Map;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;

@Properties(
    helper = "org.bytedeco.javacpp.helper.QtGui",
    inherit = QtCore.class,
    target = "org.bytedeco.javacpp.QtGui",
    value = {
        @Platform(
            include = {
                "QFont",
                "QGuiApplication"
            },
            preload = "Qt5Gui@5"
        ),
        @Platform(
            includepath = "/usr/local/Cellar/qt/5.12.0/include",
            value = "macosx-x86_64"
        )
    }
)
public class QtGui extends AbstractQtPreset {

  @Override
  public void map(InfoMap infoMap) {
    super.map(infoMap);
    infoMap
        .put(new Info("Q_GUI_EXPORT").cppTypes().annotations())

        // Line patterns
        .put(new Info("qfont.h").linePatterns(matchClass("QFont"), matchEnd()))
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
        "QBackingStore",
        "QBitmap",
        "QClipboard",
        "QCursor",
        "QFontInfo",
        "QFontMetrics",
        "QIcon",
        "QInputMethod",
        "QKeySequence",
        "QPaintEngine",
        "QPainter",
        "QPalette",
        "QPixmap",
        "QPlatformNativeInterface",
        "QRegion",
        "QScreen",
        "QStyleHints",
        "QValidator",
        "QWindow",
        "QWindowList",

        // Enums
        "QPalette::ColorRole",

        // Members
        "QFont::resolve"
    };
  }
}
