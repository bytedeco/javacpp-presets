package org.bytedeco.pytorch.data.dataframe.csv;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Robust CSV writer with RFC 4180 quoting, null tokens, optional type header, VECTOR formatting.
 */
public final class CsvWriter {
    private CsvWriter() {}

    public static void write(DataFrame df, String path) throws IOException {
        write(df, Path.of(path), CsvOptions.defaults());
    }

    public static void write(DataFrame df, String path, CsvOptions options) throws IOException {
        write(df, Path.of(path), options);
    }

    public static void write(DataFrame df, Path path, CsvOptions options) throws IOException {
        try (OutputStream out = Files.newOutputStream(path)) {
            write(df, out, options);
        }
    }

    public static void write(DataFrame df, OutputStream out, CsvOptions options) throws IOException {
        Charset cs = options.charset();
        try (Writer w = new BufferedWriter(new OutputStreamWriter(out, cs))) {
            write(df, w, options);
        }
    }

    public static void write(DataFrame df, Writer writer, CsvOptions options) throws IOException {
        BufferedWriter w = writer instanceof BufferedWriter
            ? (BufferedWriter) writer
            : new BufferedWriter(writer);

        char delim = options.delimiter();
        String nullTok = options.writeNullToken();
        CsvOptions.QuoteMode mode = options.quoteMode();
        char quote = options.quote();

        // header
        if (options.header()) {
            writeLine(w, df.columns().stream().map(Column::name).collect(Collectors.toList())
                .toArray(new String[0]), delim, quote, mode, true);
        }

        // optional type header
        if (options.typeHeader()) {
            String[] types = new String[df.columnCount()];
            for (int i = 0; i < df.columnCount(); i++) {
                Column c = df.column(i);
                types[i] = typeToken(c);
            }
            writeLine(w, types, delim, quote, mode, true);
        }

        int rows = df.rowCount();
        int cols = df.columnCount();
        for (int r = 0; r < rows; r++) {
            String[] fields = new String[cols];
            for (int c = 0; c < cols; c++) {
                Object v = df.column(c).get(r);
                fields[c] = formatValue(v, df.column(c).dtype(), nullTok);
            }
            writeLine(w, fields, delim, quote, mode, false);
        }
        w.flush();
    }

    private static String typeToken(Column c) {
        Column.DType dt = c.dtype();
        if (dt == Column.DType.VECTOR || dt == Column.DType.EMBEDDING) {
            int dim = -1;
            for (int i = 0; i < c.size(); i++) {
                Object v = c.get(i);
                if (v instanceof float[]) { dim = ((float[]) v).length; break; }
                if (v instanceof double[]) { dim = ((double[]) v).length; break; }
            }
            String base = dt == Column.DType.EMBEDDING ? "EMBEDDING" : "VECTOR";
            return dim > 0 ? base + "(" + dim + ")" : base;
        }
        return dt.name();
    }

    private static String formatValue(Object v, Column.DType dtype, String nullTok) {
        if (v == null) return nullTok;
        // Native complex cells → canonical JSON text
        if (ComplexCellCodec.isComplex(dtype) || ComplexCellCodec.isListLike(dtype)
            || ComplexCellCodec.isMapLike(dtype)
            || v instanceof float[] || v instanceof double[]
            || v instanceof int[] || v instanceof long[]
            || v instanceof boolean[]
            || v instanceof java.util.Map || v instanceof java.util.List) {
            String encoded = ComplexCellCodec.encodeText(v);
            return encoded == null ? nullTok : encoded;
        }
        return String.valueOf(v);
    }

    private static void writeLine(BufferedWriter w, String[] fields, char delim, char quote,
                                  CsvOptions.QuoteMode mode, boolean forceString) throws IOException {
        for (int i = 0; i < fields.length; i++) {
            if (i > 0) w.write(delim);
            w.write(quoteField(fields[i] == null ? "" : fields[i], delim, quote, mode, forceString));
        }
        w.write('\n');
    }

    static String quoteField(String field, char delim, char quote, CsvOptions.QuoteMode mode, boolean forceString) {
        boolean mustQuote;
        switch (mode) {
            case ALL:
                mustQuote = true;
                break;
            case NON_NUMERIC:
                mustQuote = forceString || !isPlainNumber(field);
                break;
            case MINIMAL:
            default:
                mustQuote = needsQuote(field, delim, quote);
                break;
        }
        if (!mustQuote) return field;
        StringBuilder sb = new StringBuilder(field.length() + 2);
        sb.append(quote);
        for (int i = 0; i < field.length(); i++) {
            char c = field.charAt(i);
            if (c == quote) sb.append(quote); // double the quote
            sb.append(c);
        }
        sb.append(quote);
        return sb.toString();
    }

    private static boolean needsQuote(String field, char delim, char quote) {
        if (field.isEmpty()) return false;
        if (field.charAt(0) == ' ' || field.charAt(field.length() - 1) == ' ') return true;
        for (int i = 0; i < field.length(); i++) {
            char c = field.charAt(i);
            if (c == delim || c == quote || c == '\n' || c == '\r') return true;
        }
        return false;
    }

    private static boolean isPlainNumber(String s) {
        if (s == null || s.isEmpty()) return false;
        try {
            Double.parseDouble(s);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
