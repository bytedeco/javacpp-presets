import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.llvm.clang.CXClientData;
import org.bytedeco.llvm.clang.CXCursor;
import org.bytedeco.llvm.clang.CXCursorVisitor;
import org.bytedeco.llvm.clang.CXIndex;
import org.bytedeco.llvm.clang.CXSourceRange;
import org.bytedeco.llvm.clang.CXString;
import org.bytedeco.llvm.clang.CXToken;
import org.bytedeco.llvm.clang.CXTranslationUnit;
import org.bytedeco.llvm.clang.CXUnsavedFile;
import org.bytedeco.llvm.global.clang;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import static java.lang.System.out;
import static java.util.Arrays.asList;
import static java.util.Objects.requireNonNull;
import static org.bytedeco.llvm.global.clang.CXChildVisit_Recurse;
import static org.bytedeco.llvm.global.clang.CXError_Success;
import static org.bytedeco.llvm.global.clang.CXTranslationUnit_None;
import static org.bytedeco.llvm.global.clang.clang_Cursor_getTranslationUnit;
import static org.bytedeco.llvm.global.clang.clang_createIndex;
import static org.bytedeco.llvm.global.clang.clang_disposeIndex;
import static org.bytedeco.llvm.global.clang.clang_disposeTokens;
import static org.bytedeco.llvm.global.clang.clang_disposeTranslationUnit;
import static org.bytedeco.llvm.global.clang.clang_getCursorExtent;
import static org.bytedeco.llvm.global.clang.clang_getCursorKind;
import static org.bytedeco.llvm.global.clang.clang_getCursorKindSpelling;
import static org.bytedeco.llvm.global.clang.clang_getTokenKind;
import static org.bytedeco.llvm.global.clang.clang_getTokenSpelling;
import static org.bytedeco.llvm.global.clang.clang_getTranslationUnitCursor;
import static org.bytedeco.llvm.global.clang.clang_parseTranslationUnit2;
import static org.bytedeco.llvm.global.clang.clang_tokenize;
import static org.bytedeco.llvm.global.clang.clang_visitChildren;

/**
 * Demonstrates how to avoid memory leaks when traversing the AST and tokenizing
 * ranges.
 *
 * <p>
 * This includes:
 * <ul>
 *   <li><em>Try-with-resources</em> for any {@link AutoCloseable} instances
 *   (incl. {@link Pointer} descendants);</li>
 *   <li>usage of {@link PointerScope} (see {@link #withPointerScope(Runnable)});</li>
 *   <li>invocation of {@link clang#clang_disposeIndex(CXIndex)}, {@link
 *   clang#clang_disposeTranslationUnit(CXTranslationUnit)}, and {@link
 *   clang#clang_disposeTokens(CXTranslationUnit, CXToken, int)} where
 *   appropriate.</li>
 * </ul>
 * </p>
 *
 * @see AutoCloseable
 * @see Pointer
 * @see PointerScope
 * @see #withPointerScope(Runnable)
 * @see clang#clang_disposeIndex(CXIndex)
 * @see clang#clang_disposeTranslationUnit(CXTranslationUnit)
 * @see clang#clang_disposeTokens(CXTranslationUnit, CXToken, int)
 * @author Andrey Shcheglov
 */
public final class ClangMemoryMgmtExample {
    private ClangMemoryMgmtExample() {
        assert false;
    }

    public static void main(final String... args) throws URISyntaxException {
        final URL codeUrl = requireNonNull(ClangMemoryMgmtExample.class.getResource("/sample1.cc"));
        final Path absoluteFile = Paths.get(codeUrl.toURI()).toAbsolutePath().normalize();
        parse(absoluteFile, asList("-std=gnu++20", "-fparse-all-comments"));
    }

    private static void parse(
            final Path absoluteFile,
            final List<String> commandLineArgs
    ) {
        withPointerScope(() -> {
            withIndex(clang_createIndex(1, 0), index -> {
                withTranslationUnit(CXTranslationUnit::new, translationUnit -> {
                    try (final BytePointer sourceFilename = new BytePointer(absoluteFile.toString())) {
                        try (final PointerPointer<Pointer> commandLineArgsPtr = new PointerPointer<>(commandLineArgs.toArray(new String[0]))) {
                            try (final CXUnsavedFile unsavedFiles = new CXUnsavedFile()) {
                                final int unsavedFilesCount = 0;

                                final int errorCode = clang_parseTranslationUnit2(
                                        index,
                                        sourceFilename,
                                        commandLineArgsPtr,
                                        commandLineArgs.size(),
                                        unsavedFiles,
                                        unsavedFilesCount,
                                        CXTranslationUnit_None,
                                        translationUnit
                                );

                                if (errorCode == CXError_Success) {
                                    try (final CXCursor rootCursor = clang_getTranslationUnitCursor(translationUnit)) {
                                        clang_visitChildren(rootCursor, new AstVisitor(), null);
                                    }
                                } else {
                                    out.printf("Failed to parse %s; parser returned code %d%n", absoluteFile, errorCode);
                                }
                            }
                        }
                    }
                });
            });
        });
    }

    private static void withPointerScope(final Runnable block) {
        try (final PointerScope ignored = new PointerScope()) {
            block.run();
        }
    }

    private static void withIndex(
            final CXIndex index,
            final Consumer<CXIndex> block
    ) {
        try (final CXIndex ignored = index) {
            try {
                block.accept(index);
            } finally {
                clang_disposeIndex(index);
            }
        }
    }

    private static void withTranslationUnit(
            final Supplier<CXTranslationUnit> lazyTranslationUnit,
            final Consumer<CXTranslationUnit> block
    ) {
        try (final CXTranslationUnit translationUnit = lazyTranslationUnit.get()) {
            try {
                block.accept(translationUnit);
            } finally {
                clang_disposeTranslationUnit(translationUnit);
            }
        }
    }

    private static void forEachToken(
            final CXCursor cursor,
            final Consumer<? super CXToken> action
    ) {
        try (final CXSourceRange extent = clang_getCursorExtent(cursor)) {
            try (final CXTranslationUnit translationUnit = clang_Cursor_getTranslationUnit(cursor)) {
                try (final CXToken tokens = new CXToken()) {
                    final int[] tokenCountRef = new int[1];
                    clang_tokenize(translationUnit, extent, tokens, tokenCountRef);
                    final int tokenCount = tokenCountRef[0];
                    try {
                        IntStream.range(0, tokenCount)
                             .mapToObj(tokens::position)
                             .forEach(action);
                    } finally {
                        tokens.position(0L);
                        clang_disposeTokens(translationUnit, tokens, tokenCount);
                    }
                }
            }
        }
    }

    private static final class AstVisitor extends CXCursorVisitor {
        @Override
        public int call(final CXCursor cursor, final CXCursor parent, final CXClientData clientData) {
            try (final CXCursor c = cursor; final CXCursor p = parent; final CXClientData d = clientData) {
                /*-
                 * Entering a new `PointerScope` here is 100% necessary,
                 * probably because the outer ("lower") stack frame is a
                 * native one (i.e. `call()` is directly invoked by the
                 * native code).
                 *
                 * Despite previously registered ("outer") pointer scopes
                 * are still visible, having only a single scope per
                 * translation unit (i.e., AST tree) rather than per cursor
                 * eventually results in 100% usage of all CPU cores -- in
                 * the native code.
                 */
                withPointerScope(() -> {
                    try (final CXString spelling = clang_getCursorKindSpelling(clang_getCursorKind(cursor))) {
                        out.println(spelling.getString());
                    }

                    try (final CXTranslationUnit translationUnit = clang_Cursor_getTranslationUnit(cursor)) {
                        forEachToken(cursor, token -> {
                            final TokenKind kind = TokenKind.valueOf(clang_getTokenKind(token));
                            try (final CXString spelling = clang_getTokenSpelling(translationUnit, token)) {
                                out.printf("\t%s(\"%s\")%n", kind, spelling.getString());
                            }
                        });
                    }

                });
            }

            return CXChildVisit_Recurse;
        }
    }

    private enum TokenKind {
        Punctuation,

        Keyword,

        Identifier,

        Literal,

        Comment,
        ;

        private static TokenKind valueOf(final int ordinal) {
            return values()[ordinal];
        }
    }
}
