package org.bytedeco.javacpp.presets;

import java.util.HashMap;
import java.util.Map;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;

@Properties(
    helper = "org.bytedeco.javacpp.helper.QtCore",
    target = "org.bytedeco.javacpp.QtCore",
    value = {
        @Platform(
            include = {
                "qlogging.h",
                "qnamespace.h",
                "QAbstractEventDispatcher",
                "QCoreApplication",
                "QCoreEvent",
                "QEventLoop",
                "QObject",
                "QSize",
                "QString",
                "QStringList",
                "QThread"
            },
            preload = "Qt5Core@5"
        ),
        @Platform(
            includepath = "/usr/local/Cellar/qt/5.12.0/include",
            value = "macosx-x86_64"
        )
    }
)
public class QtCore extends AbstractQtPreset {

  @Override
  public void map(InfoMap infoMap) {
    super.map(infoMap);
    infoMap
        .put(new Info("Q_CORE_EXPORT", "Q_INVOKABLE").cppTypes().annotations())
        .put(new Info("QT_DEPRECATED").cppTypes().annotations("@Deprecated"))

        // Helper classes
        .put(new Info("QString").base("AbstractQString"))

        // Line patterns
        .put(new Info("qabstracteventdispatcher.h").linePatterns(
            matchClass("QAbstractEventDispatcher"), matchEnd()
        ))
        .put(new Info("qcoreapplication.h").linePatterns(
            matchClass("QCoreApplication"), matchEnd()
        ))
        .put(new Info("qcoreevent.h").linePatterns(matchClass("QEvent"), matchEnd()))
        .put(new Info("qeventloop.h").linePatterns(matchClass("QEventLoop"), matchEnd()))
        .put(new Info("qlogging.h").linePatterns(
            matchEnum("QtMsgType"), "",
            matchClass("QMessageLogContext"), matchEnd(),
            matchClass("QMessageLogger"), " *void critical\\(.*",
            " *void fatal\\(.*", matchEnd(),
            "typedef .*\\bQtMessageHandler\\b.*", "\\bqInstallMessageHandler\\(.*"
        ))
        .put(new Info("qnamespace.h").linePatterns("#ifndef Q_MOC_RUN", "}"))
        .put(new Info("qobject.h").linePatterns(matchClass("QObject"), matchEnd()))
        .put(new Info("qsize.h").linePatterns(matchClass("QSize"), matchEnd()))
        .put(new Info("qstring.h").linePatterns(
            matchClass("QString"), matchEnd()
        ))
        .put(new Info("qstringlist.h").linePatterns("#ifdef Q_QDOC", matchEnd()))
        .put(new Info("qthread.h").linePatterns(matchClass("QThread"), matchEnd()))

        // Methods
        .put(new Info("QString::toStdString")
            .javaText("public native @StdString String toStdString();"));
  }

  @Override
  protected String[] defineFalse() {
    return new String[]{
        "QT_DEPRECATED_SINCE(5,0)",
        "QT_DEPRECATED_SINCE(5, 0)",
        "QT_DEPRECATED_SINCE(5, 9)",

        // qnamespace.h
        "defined(Q_COMPILER_CLASS_ENUM) && defined(Q_COMPILER_CONSTEXPR)",
        "Q_MOC_RUN",

        // QObject
        "QT_HAS_INCLUDE(<chrono>)",

        // QString
        "defined(Q_COMPILER_REF_QUALIFIERS) && !defined(QT_COMPILING_QSTRING_COMPAT_CPP) && !defined(Q_CLANG_QDOC)",
        "defined(Q_COMPILER_UNICODE_STRINGS)",
        "!defined(QT_NO_CAST_FROM_ASCII) && !defined(QT_RESTRICTED_CAST_FROM_ASCII)",
        "defined(Q_OS_DARWIN) || defined(Q_QDOC)",
        "defined(Q_STDLIB_UNICODE_STRINGS) || defined(Q_QDOC)"
    };
  }

  @Override
  protected String[] defineTrue() {
    return new String[]{
        "Q_CLANG_QDOC",
        "QT_CONFIG(menu)",
        "QT_NO_DEBUG_STREAM",
        "Q_QDOC"
    };
  }

  @Override
  protected String[] enums() {
    return new String[]{
        "QEventLoop::ProcessEventsFlag",
        "QString::SectionFlag",
        "Qt::ApplicationAttribute",
        "Qt::CaseSensitivity",
        "Qt::FindChildOption",
        "Qt::Orientation",
        "QtMsgType"
    };
  }

  @Override
  protected Map<String, String> flags() {
    Map<String, String> flags = new HashMap<>();
    flags.put("QEventLoop::ProcessEventsFlags", "QEventLoop::ProcessEventsFlag");
    flags.put("QString::SectionFlags", "QString::SectionFlag");
    return flags;
  }

  @Override
  protected Map<String, String> macros() {
    Map<String, String> macros = new HashMap<>();

    // Types
    macros.put("char16_t", "unsigned int");
    macros.put("char32_t", "unsigned long");
    macros.put("qint8", "signed char");
    macros.put("quint8", "unsigned char");
    macros.put("qint16", "short");
    macros.put("quint16", "unsigned short");
    macros.put("qint32", "int");
    macros.put("quint32", "unsigned int");
    macros.put("qint64", "long long");
    macros.put("quint64", "unsigned long long");
    macros.put("qlonglong", "long long");
    macros.put("qulonglong", "unsigned long long");
    macros.put("uchar", "unsigned char");
    macros.put("ushort", "unsigned short");
    macros.put("uint", "unsigned int");
    macros.put("ulong", "unsigned long");
    macros.put("qreal", "double");

    // Macros
    macros.put("Q_ALWAYS_INLINE", "inline");
    macros.put("Q_DECL_COLD_FUNCTION", "");
    macros.put("Q_DECL_CONSTEXPR", "");
    macros.put("Q_DECL_DEPRECATED", "");
    macros.put("Q_DECL_NOTHROW", "");
    macros.put("Q_DECL_RELAXED_CONSTEXPR", "");
    macros.put("Q_DECL_UNUSED", "");
    macros.put("Q_DECLARE_FLAGS", "#define Q_DECLARE_FLAGS(arg0, arg1)");
    macros.put("Q_DECLARE_OPERATORS_FOR_FLAGS(arg0)", "");
    macros.put("Q_DECLARE_PRIVATE(arg0)", "");
    macros.put("Q_DECLARE_SHARED(arg0)", "");
    macros.put("Q_DISABLE_COPY(arg0)", "");
    macros.put("Q_ENUM(arg0)", "");
    macros.put("Q_ENUMS(arg0)", "");
    macros.put("Q_FLAG(arg0)", "");
    macros.put("Q_GADGET", "");
    macros.put("Q_NEVER_INLINE", "");
    macros.put("Q_NORETURN", "");
    macros.put("Q_OBJECT", "");
    macros.put("Q_PROPERTY(arg0)", "");
    macros.put("Q_REQUIRED_RESULT", "");
    macros.put("Q_SIGNALS", "private");
    macros.put("Q_SLOTS", "");
    macros.put("QDOC_PROPERTY(arg0)", "");
    macros.put("QT_VERSION", "0");
    macros.put("QT6_VIRTUAL", "virtual");

    return macros;
  }

  @Override
  protected String[] skip() {
    return new String[]{
        // Classes
        "QAbstractNativeEventFilter",
        "QByteArray",
        "QChar",
        "QCharRef",
        "QEvent",
        "QLatin1String",
        "QList",
        "QLocale",
        "QLoggingCategory",
        "QMargins",
        "QMetaMethod",
        "QMetaObject::Connection",
        "QObjectList",
        "QObjectUserData",
        "QPoint",
        "QRect",
        "QRegExp",
        "QRegularExpression",
        "QRegularExpressionMatch",
        "QSocketNotifier",
        "QStringDataPtr",
        "QStringList",
        "QStringRef",
        "QStringView",
        "QThread",
        "QTranslator",
        "QVariant",
        "QVector",
        "QWinEventNotifier",

        // Enums
        "QChar::SpecialCharacter",
        "QChar::UnicodeVersion",

        // Methods
        "QObject::property",

        // Types
        "const_iterator",
        "const_reverse_iterator",
        "iterator",
        "reverse_iterator",
        "std::reverse_iterator"
    };
  }
}
