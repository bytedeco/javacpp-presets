/*
 * Copyright (C) 2019-2020 Greg Hart, Samuel Audet
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
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;

@Properties(
    inherit = javacpp.class,
    target = "org.bytedeco.qt.Qt5Core",
    global = "org.bytedeco.qt.global.Qt5Core",
    helper = "org.bytedeco.qt.helper.Qt5Core",
    value = {
        @Platform(
            include = {
                "./qtcore.h",
                "qcoreevent.h",
                "qlogging.h",
                "qnamespace.h",
                "QAbstractEventDispatcher",
                "QByteArray",
                "QCoreApplication",
                "QCoreEvent",
                "QEventLoop",
                "QObject",
                "QSize",
                "QString",
                "QStringList",
                "QThread",
                "QVariant"
            },
            link = "Qt5Core@.5"
        )
    }
)
public class Qt5Core extends QtInfoMapper {
    static { Loader.checkVersion("org.bytedeco", "qt"); }

  @Override
  public void map(InfoMap infoMap) {
    super.map(infoMap);
    infoMap
        .put(new Info("Q_CORE_EXPORT", "Q_INVOKABLE").cppTypes().annotations())
        .put(new Info("QT_ASCII_CAST_WARN", "QT_DEPRECATED", "QT_DEPRECATED_VERSION_5_14").cppTypes().annotations("@Deprecated"))

        // Helper classes
        .put(new Info("QString").base("AbstractQString").virtualize())

        // Line patterns
        .put(new Info("qabstracteventdispatcher.h").linePatterns(
            matchClass("QAbstractEventDispatcher"), matchEnd()
        ))
        .put(new Info("qbytearray.h").linePatterns(matchClass("QByteArray"), matchEnd()))
        .put(new Info("qcoreapplication.h").linePatterns(
            matchClass("QCoreApplication"), matchEnd()
        ))
        .put(new Info("qcoreevent.h").linePatterns("class Q_CORE_EXPORT .*", "QT_END_NAMESPACE"))
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
        .put(new Info("qvariant.h").linePatterns(
            matchClass("QVariant"), ".*\\*typeToName\\(.*",
            " *typedef Private DataPtr;", matchEnd()))

        // Members
        .put(new Info("QVariant::toString").javaNames("toQString"))

        // Methods
        .put(new Info("QObject::event")
            .javaText("@Virtual protected native @Cast(\"bool\") boolean event(QEvent event);"))
        .put(new Info("QObject::eventFilter")
            .javaText("@Virtual protected native @Cast(\"bool\") boolean eventFilter(QObject watched, QEvent event);"))
        .put(new Info("QString::toStdString")
            .javaText("public native @StdString String toStdString();"));
  }

  @Override
  protected String[] defineFalse() {
    return new String[]{
        "QT_DEPRECATED_SINCE(5,0)",
        "QT_DEPRECATED_SINCE(5, 0)",
        "QT_DEPRECATED_SINCE(5, 9)",
        "QT_DEPRECATED_SINCE(5, 13)",
        "QT_DEPRECATED_SINCE(5, 15)",

        // qnamespace.h
        "defined(Q_COMPILER_CLASS_ENUM) && defined(Q_COMPILER_CONSTEXPR)",
        "defined(Q_COMPILER_CONSTEXPR)",
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
        "QByteArray::Base64Option",
        "QEvent::Type",
        "QEventLoop::ProcessEventsFlag",
        "QSizePolicy::ControlType",
        "QString::SectionFlag",
        "Qt::ApplicationAttribute",
        "Qt::CaseSensitivity",
        "Qt::FindChildOption",
        "Qt::GestureFlag",
        "Qt::InputMethodHint",
        "Qt::MouseButton",
        "Qt::Orientation",
        "Qt::WindowState",
        "QtMsgType"
    };
  }

  @Override
  protected Map<String, String> flags() {
    Map<String, String> flags = new HashMap<>();
    flags.put("QByteArray::Base64Options", "QByteArray::Base64Option");
    flags.put("QEventLoop::ProcessEventsFlags", "QEventLoop::ProcessEventsFlag");
    flags.put("QSizePolicy::ControlTypes", "QSizePolicy::ControlType");
    flags.put("QString::SectionFlags", "QString::SectionFlag");
    flags.put("Qt::FindChildOptions", "Qt::FindChildOption");
    flags.put("Qt::GestureFlags", "Qt::GestureFlag");
    flags.put("Qt::InputMethodHints", "Qt::InputMethodHint");
    flags.put("Qt::MouseButtons", "Qt::MouseButton");
    flags.put("Qt::Orientations", "Qt::Orientation");
    flags.put("Qt::WindowStates", "Qt::WindowState");
    return flags;
  }

  @Override
  protected String[] intEnums() {
    return new String[]{
        "Qt::AlignmentFlag",
        "Qt::KeyboardModifier",
        "Qt::MatchFlag",
        "Qt::TextInteractionFlag",
        "Qt::WindowType",
        "QSizePolicy::Policy"
    };
  }

  @Override
  protected String[] intFlags() {
    return new String[]{
        "Qt::Alignment",
        "Qt::KeyboardModifiers",
        "Qt::MatchFlags",
        "Qt::TextInteractionFlags",
        "Qt::WindowFlags"
    };
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
    macros.put("qintptr", "long int");
    macros.put("uchar", "unsigned char");
    macros.put("ushort", "unsigned short");
    macros.put("uint", "unsigned int");
    macros.put("ulong", "unsigned long");
    macros.put("qreal", "double");
    macros.put("milliseconds", "long long");

    // Macros
    macros.put("Q_ALWAYS_INLINE", "inline");
    macros.put("Q_DECL_COLD_FUNCTION", "");
    macros.put("Q_DECL_CONSTEXPR", "");
    macros.put("Q_DECL_ENUMERATOR_DEPRECATED_X", "");
    macros.put("Q_DECL_ENUMERATOR_DEPRECATED", "");
    macros.put("Q_DECL_DEPRECATED", "");
    macros.put("Q_DECL_NOEXCEPT", "");
    macros.put("Q_DECL_NOTHROW", "");
    macros.put("Q_DECL_RELAXED_CONSTEXPR", "");
    macros.put("Q_DECL_UNUSED", "");
    macros.put("Q_DECLARE_FLAGS(arg0, arg1)", "");
    macros.put("Q_DUMMY_COMPARISON_OPERATOR(arg0)", "");
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
        "QAbstractItemModel",
        "QAbstractNativeEventFilter",
        "QAtomicInt",
        "QBitArray",
        "QByteArrayDataPtr",
        "QByteRef",
        "QChar",
        "QCharRef",
        "QDataStream",
        "QDate",
        "QDateTime",
        "QEasingCurve",
        "QFunctionPointer",
        "QHash",
        "QJsonArray",
        "QJsonDocument",
        "QJsonObject",
        "QJsonValue",
        "QLatin1String",
        "QLine",
        "QLineF",
        "QList",
        "QLocale",
        "QLoggingCategory",
        "QMap",
        "QMargins",
        "QMetaMethod",
        "QMetaObject::Connection",
        "QMimeData",
        "QModelIndex",
        "QObjectList",
        "QObjectUserData",
        "QPersistentModelIndex",
        "QPostEventList",
        "QPoint",
        "QPointF",
        "QRect",
        "QRectF",
        "QRegExp",
        "QRegularExpression",
        "QRegularExpressionMatch",
        "QSizeF",
        "QSocketNotifier",
        "QStringDataPtr",
        "QStringList",
        "QStringRef",
        "QStringView",
        "QThread",
        "QTime",
        "QTranslator",
        "QUrl",
        "QUuid",
        "QVector",
        "QWinEventNotifier",

        // Enums
        "QChar::SpecialCharacter",
        "QChar::UnicodeVersion",
        "QVariant::Type",

        // Files
        "qtcore.h",

        // Methods
        "QObject::property",
        "QVariant::typeToName",
        "QVariant::data_ptr",

        // Types
        "const_iterator",
        "const_reverse_iterator",
        "iterator",
        "QByteArray::const_iterator",
        "QByteArray::iterator",
        "QVariant::Handler",
        "QVariant::Private",
        "QVariant::PrivateShared",
        "reverse_iterator",
        "std::reverse_iterator"
    };
  }

  @Override
  protected String[] virtual() {
    return new String[]{
//        "QAbstractEventDispatcher",
        "QByteArray",
        "QCoreApplication",
        "QCoreEvent",
        "QEventLoop",
        "QObject",
        "QSize",
//        "QString",
        "QStringList",
        "QThread",
        "QVariant"
    };
  }
}
