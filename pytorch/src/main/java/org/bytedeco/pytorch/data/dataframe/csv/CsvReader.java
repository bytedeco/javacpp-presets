package org.bytedeco.pytorch.data.dataframe.csv;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.*;

/**
 * Production-oriented CSV reader with RFC 4180 quoting, multi-line fields,
 * type inference, optional type-header row, null tokens, comments, and BOM strip.
 */
public final class CsvReader {
    private CsvReader() {}

    public static DataFrame read(String path) throws IOException {
        return read(Path.of(path), CsvOptions.defaults());
    }

    public static DataFrame read(String path, CsvOptions options) throws IOException {
        return read(Path.of(path), options);
    }

    public static DataFrame read(Path path, CsvOptions options) throws IOException {
        try (InputStream in = Files.newInputStream(path)) {
            return read(in, options);
        }
    }

    public static DataFrame read(InputStream in, CsvOptions options) throws IOException {
        Charset cs = options.charset();
        // strip UTF-8 BOM if present
        PushbackInputStream pin = new PushbackInputStream(in, 3);
        byte[] bom = new byte[3];
        int n = pin.read(bom);
        boolean hasBom = options.stripBom()
            && n == 3
            && (bom[0] & 0xFF) == 0xEF
            && (bom[1] & 0xFF) == 0xBB
            && (bom[2] & 0xFF) == 0xBF;
        if (n > 0 && !hasBom) {
            pin.unread(bom, 0, n);
        } else if (n > 0 && hasBom) {
            // BOM consumed
        } else if (n > 0) {
            pin.unread(bom, 0, n);
        }
        try (Reader r = new InputStreamReader(pin, cs)) {
            return read(r, options);
        }
    }

    public static DataFrame read(Reader reader, CsvOptions options) throws IOException {
        Parser parser = new Parser(reader, options);
        List<String[]> rawRows = new ArrayList<>();
        long physicalLine = 0;

        // skip leading rows
        for (int i = 0; i < options.skipRows(); i++) {
            String[] skipped = parser.nextRecord();
            if (skipped == null) break;
            physicalLine = parser.lineNumber();
        }

        String[] headers = null;
        Column.DType[] forcedTypes = null;
        int[] vectorDims = null;

        if (options.columnNames() != null && !options.columnNames().isEmpty()) {
            headers = options.columnNames().toArray(new String[0]);
        }

        if (options.header()) {
            String[] headerRow = parser.nextRecord();
            if (headerRow == null) return DataFrame.create();
            if (headers == null) headers = sanitizeHeaders(headerRow);

            // type header: explicit or auto-detect
            boolean wantTypeHeader = options.typeHeader();
            String[] maybeType = parser.peekRecord();
            if (maybeType != null) {
                if (!wantTypeHeader && looksLikeTypeHeader(maybeType, headers.length)) {
                    wantTypeHeader = true;
                }
                if (wantTypeHeader) {
                    String[] typeRow = parser.nextRecord();
                    forcedTypes = new Column.DType[headers.length];
                    vectorDims = new int[headers.length];
                    for (int i = 0; i < headers.length; i++) {
                        String tok = i < typeRow.length ? typeRow[i].trim() : "STRING";
                        forcedTypes[i] = parseTypeToken(tok);
                        vectorDims[i] = parseVectorDim(tok);
                    }
                }
            }
        }

        // read data rows (up to maxRows; also keep sample for inference)
        int max = options.maxRows();
        while (true) {
            if (max >= 0 && rawRows.size() >= max) break;
            String[] rec = parser.nextRecord();
            if (rec == null) break;
            // comment already filtered inside parser for full-line comments
            rawRows.add(rec);
        }

        if (headers == null) {
            int cols = 0;
            for (String[] r : rawRows) cols = Math.max(cols, r.length);
            headers = new String[cols];
            for (int i = 0; i < cols; i++) headers[i] = "col_" + i;
        }

        int numCols = headers.length;

        // resolve dtypes
        Column.DType[] dtypes = new Column.DType[numCols];
        if (options.schema() != null) {
            for (int i = 0; i < numCols; i++) {
                Column.DType t = options.schema().get(headers[i]);
                dtypes[i] = t != null ? t : Column.DType.STRING;
            }
        } else if (forcedTypes != null) {
            System.arraycopy(forcedTypes, 0, dtypes, 0, numCols);
        } else if (options.inferSchema()) {
            dtypes = inferTypes(rawRows, numCols, options, Math.min(options.inferSampleSize(), rawRows.size()));
        } else {
            Arrays.fill(dtypes, Column.DType.STRING);
        }

        DataFrame df = DataFrame.create();
        for (int i = 0; i < numCols; i++) {
            df.addColumn(headers[i], dtypes[i]);
        }

        for (int ri = 0; ri < rawRows.size(); ri++) {
            String[] row = rawRows.get(ri);
            if (options.strict() && row.length != numCols) {
                throw new CsvParseException(
                    "Ragged row: expected " + numCols + " fields, got " + row.length,
                    parser.lineNumber(), -1, Arrays.toString(row));
            }
            Object[] values = new Object[numCols];
            for (int ci = 0; ci < numCols; ci++) {
                String raw = ci < row.length ? row[ci] : "";
                values[ci] = parseValue(raw, dtypes[ci], options,
                    vectorDims != null ? vectorDims[ci] : -1,
                    parser.lineNumber(), ci);
            }
            try {
                df.addRow(values);
            } catch (IllegalArgumentException ex) {
                // fall back cell-by-cell for safety
                int idx = df.addEmptyRow();
                for (int ci = 0; ci < numCols; ci++) {
                    df.set(idx, headers[ci], values[ci]);
                }
            }
        }
        return df;
    }

    // ---- type inference ----

    private static Column.DType[] inferTypes(List<String[]> rows, int numCols, CsvOptions opt, int sample) {
        Column.DType[] out = new Column.DType[numCols];
        for (int c = 0; c < numCols; c++) {
            boolean canBool = true, canLong = true, canDouble = true, canDate = true, canDateTime = true, canVector = true;
            boolean canList = true, canMap = true;
            int nonNull = 0;
            int limit = Math.min(sample, rows.size());
            for (int r = 0; r < limit; r++) {
                String[] row = rows.get(r);
                String v = c < row.length ? row[c] : "";
                if (opt.isNullToken(v)) continue;
                nonNull++;
                String t = v.trim();
                if (canBool && !isBoolean(t)) canBool = false;
                if (canLong && !isLong(t)) canLong = false;
                if (canDouble && !isDouble(t)) canDouble = false;
                if (canDate && !isDate(t)) canDate = false;
                if (canDateTime && !isDateTime(t)) canDateTime = false;
                if (canVector && !isVector(t)) canVector = false;
                // JSON array that is not a pure numeric vector → LIST
                if (canList) {
                    if (!(t.startsWith("[") && t.endsWith("]"))) canList = false;
                    else if (isVector(t)) { /* numeric vector preferred */ }
                    else {
                        try { ComplexCellCodec.decodeText(t, Column.DType.LIST); }
                        catch (Exception e) { canList = false; }
                    }
                }
                if (canMap) {
                    if (!(t.startsWith("{") && t.endsWith("}"))) canMap = false;
                    else {
                        try { ComplexCellCodec.decodeText(t, Column.DType.MAP); }
                        catch (Exception e) { canMap = false; }
                    }
                }
            }
            if (nonNull == 0) {
                out[c] = Column.DType.STRING;
            } else if (canBool) {
                out[c] = Column.DType.BOOLEAN;
            } else if (canLong) {
                out[c] = Column.DType.INT64;
            } else if (canDouble) {
                out[c] = Column.DType.FLOAT64;
            } else if (canDateTime) {
                out[c] = Column.DType.DATETIME;
            } else if (canDate) {
                out[c] = Column.DType.DATE;
            } else if (canVector) {
                out[c] = Column.DType.VECTOR;
            } else if (canList) {
                out[c] = Column.DType.LIST;
            } else if (canMap) {
                out[c] = Column.DType.MAP;
            } else {
                out[c] = Column.DType.STRING;
            }
        }
        return out;
    }

    private static boolean isBoolean(String s) {
        return "true".equalsIgnoreCase(s) || "false".equalsIgnoreCase(s)
            || "1".equals(s) || "0".equals(s) || "yes".equalsIgnoreCase(s) || "no".equalsIgnoreCase(s);
    }

    private static boolean isLong(String s) {
        try {
            Long.parseLong(s);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private static boolean isDouble(String s) {
        try {
            if (s.isEmpty()) return false;
            Double.parseDouble(s);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private static final DateTimeFormatter[] DATE_FMTS = {
        DateTimeFormatter.ISO_LOCAL_DATE,
        DateTimeFormatter.ofPattern("yyyy/MM/dd"),
        DateTimeFormatter.ofPattern("MM/dd/yyyy"),
        DateTimeFormatter.ofPattern("dd-MM-yyyy")
    };
    private static final DateTimeFormatter[] DATETIME_FMTS = {
        DateTimeFormatter.ISO_LOCAL_DATE_TIME,
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
        DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss")
    };

    private static boolean isDate(String s) {
        for (DateTimeFormatter f : DATE_FMTS) {
            try { LocalDate.parse(s, f); return true; } catch (DateTimeParseException ignored) {}
        }
        return false;
    }

    private static boolean isDateTime(String s) {
        for (DateTimeFormatter f : DATETIME_FMTS) {
            try { LocalDateTime.parse(s, f); return true; } catch (DateTimeParseException ignored) {}
        }
        return false;
    }

    private static boolean isVector(String s) {
        String t = s.trim();
        if (!(t.startsWith("[") && t.endsWith("]"))) return false;
        String inner = t.substring(1, t.length() - 1).trim();
        if (inner.isEmpty()) return true;
        String[] parts = inner.split("[,;\\s]+");
        for (String p : parts) {
            if (p.isEmpty()) continue;
            if (!isDouble(p)) return false;
        }
        return true;
    }

    // ---- value parse ----

    private static Object parseValue(String raw, Column.DType dtype, CsvOptions opt,
                                     int vectorDim, long line, int field) {
        if (opt.isNullToken(raw)) return null;
        String s = raw.trim();
        try {
            switch (dtype) {
                case INT32: return Integer.parseInt(s);
                case INT64: return Long.parseLong(s);
                case FLOAT32: return Float.parseFloat(s);
                case FLOAT64: return Double.parseDouble(s);
                case BOOLEAN:
                    if ("1".equals(s) || "yes".equalsIgnoreCase(s) || "true".equalsIgnoreCase(s)) return true;
                    if ("0".equals(s) || "no".equalsIgnoreCase(s) || "false".equalsIgnoreCase(s)) return false;
                    return Boolean.parseBoolean(s);
                case DATE:
                    for (DateTimeFormatter f : DATE_FMTS) {
                        try { return LocalDate.parse(s, f); } catch (DateTimeParseException ignored) {}
                    }
                    throw new CsvParseException("Cannot parse DATE", line, field, s);
                case DATETIME:
                    for (DateTimeFormatter f : DATETIME_FMTS) {
                        try { return LocalDateTime.parse(s, f); } catch (DateTimeParseException ignored) {}
                    }
                    throw new CsvParseException("Cannot parse DATETIME", line, field, s);
                case VECTOR:
                case EMBEDDING:
                    return parseVector(s, vectorDim, line, field);
                case LIST:
                case MAP:
                case STRUCT:
                case JSON:
                    return ComplexCellCodec.decodeText(s, dtype);
                case STRING:
                default:
                    // Auto-detect JSON nested cells when stored as text without type header
                    if (s != null) {
                        String t = s.trim();
                        if ((t.startsWith("[") && t.endsWith("]")) || (t.startsWith("{") && t.endsWith("}"))) {
                            try {
                                return ComplexCellCodec.decodeText(t, ComplexCellCodec.inferComplex(t));
                            } catch (Exception ignored) { /* plain string */ }
                        }
                    }
                    return raw; // keep original (no trim for strings)
            }
        } catch (CsvParseException e) {
            if (opt.strict()) throw e;
            return raw;
        } catch (Exception e) {
            if (opt.strict()) throw new CsvParseException("Parse failed for " + dtype, line, field, s, e);
            return raw;
        }
    }

    private static float[] parseVector(String s, int expectedDim, long line, int field) {
        String t = s.trim();
        if (t.startsWith("[") && t.endsWith("]")) t = t.substring(1, t.length() - 1).trim();
        if (t.isEmpty()) return new float[0];
        String[] parts = t.split("[,;\\s]+");
        List<Float> vals = new ArrayList<>(parts.length);
        for (String p : parts) {
            if (p.isEmpty()) continue;
            vals.add(Float.parseFloat(p));
        }
        if (expectedDim > 0 && vals.size() != expectedDim) {
            throw new CsvParseException("VECTOR dim mismatch: expected " + expectedDim + " got " + vals.size(),
                line, field, s);
        }
        float[] out = new float[vals.size()];
        for (int i = 0; i < vals.size(); i++) out[i] = vals.get(i);
        return out;
    }

    // ---- headers / type header ----

    private static String[] sanitizeHeaders(String[] raw) {
        String[] h = new String[raw.length];
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < raw.length; i++) {
            String name = raw[i] == null || raw[i].trim().isEmpty() ? "col_" + i : raw[i].trim();
            String base = name;
            int n = 1;
            while (!seen.add(name)) {
                name = base + "_" + (n++);
            }
            h[i] = name;
        }
        return h;
    }

    private static boolean looksLikeTypeHeader(String[] tokens, int expectedCols) {
        if (tokens.length != expectedCols && expectedCols > 0) return false;
        if (tokens.length == 0) return false;
        int hits = 0;
        for (String t : tokens) {
            if (t == null) return false;
            String u = t.trim().toUpperCase(Locale.ROOT);
            if (u.isEmpty()) return false;
            if (u.startsWith("VECTOR") || u.startsWith("EMBEDDING")
                || u.equals("INT32") || u.equals("INT64") || u.equals("INT") || u.equals("LONG")
                || u.equals("FLOAT32") || u.equals("FLOAT64") || u.equals("FLOAT") || u.equals("DOUBLE")
                || u.equals("BOOLEAN") || u.equals("BOOL")
                || u.equals("STRING") || u.equals("STR") || u.equals("UTF8")
                || u.equals("DATE") || u.equals("DATETIME") || u.equals("TIMESTAMP")
                || u.equals("TIME") || u.equals("DURATION") || u.equals("TENSOR")) {
                hits++;
            }
        }
        return hits == tokens.length;
    }

    private static Column.DType parseTypeToken(String tok) {
        String u = tok.trim().toUpperCase(Locale.ROOT);
        int paren = u.indexOf('(');
        if (paren > 0) u = u.substring(0, paren);
        switch (u) {
            case "INT32": case "INT": return Column.DType.INT32;
            case "INT64": case "LONG": return Column.DType.INT64;
            case "FLOAT32": case "FLOAT": return Column.DType.FLOAT32;
            case "FLOAT64": case "DOUBLE": return Column.DType.FLOAT64;
            case "BOOLEAN": case "BOOL": return Column.DType.BOOLEAN;
            case "DATE": return Column.DType.DATE;
            case "DATETIME": case "TIMESTAMP": return Column.DType.DATETIME;
            case "TIME": return Column.DType.TIME;
            case "DURATION": return Column.DType.DURATION;
            case "TENSOR": return Column.DType.TENSOR;
            case "VECTOR": return Column.DType.VECTOR;
            case "EMBEDDING": return Column.DType.EMBEDDING;
            case "LIST": case "LIST_VIEW": return Column.DType.LIST;
            case "MAP": case "MAP_VIEW": case "DICT": return Column.DType.MAP;
            case "STRUCT": case "RECORD": return Column.DType.STRUCT;
            case "JSON": return Column.DType.JSON;
            case "BINARY": case "BYTES": return Column.DType.BINARY;
            default: return Column.DType.STRING;
        }
    }

    private static int parseVectorDim(String tok) {
        int a = tok.indexOf('(');
        int b = tok.indexOf(')');
        if (a > 0 && b > a) {
            try { return Integer.parseInt(tok.substring(a + 1, b).trim()); }
            catch (Exception ignored) {}
        }
        return -1;
    }

    // ---- record parser (RFC 4180 + multi-line) ----

    static final class Parser {
        private final Reader in;
        private final CsvOptions opt;
        private final char delim;
        private final char quote;
        private final char escape;
        private final Character comment;
        private long lineNumber = 0;
        private int nextChar = -2; // peek buffer; -2 = empty
        private String[] peeked = null;

        Parser(Reader in, CsvOptions opt) {
            this.in = in instanceof BufferedReader ? in : new BufferedReader(in, 64 * 1024);
            this.opt = opt;
            this.delim = opt.delimiter();
            this.quote = opt.quote();
            this.escape = opt.escape();
            this.comment = opt.comment();
        }

        long lineNumber() { return lineNumber; }

        String[] peekRecord() throws IOException {
            if (peeked == null) peeked = nextRecord();
            return peeked;
        }

        String[] nextRecord() throws IOException {
            if (peeked != null) {
                String[] r = peeked;
                peeked = null;
                return r;
            }
            while (true) {
                String[] rec = readOneRecord();
                if (rec == null) return null;
                // skip full-line comments (only if first field starts with comment and single-field-ish)
                if (comment != null && rec.length > 0 && rec[0] != null
                    && !rec[0].isEmpty() && rec[0].charAt(0) == comment) {
                    // treat as comment line only when the record is effectively a comment line
                    // (first char of first field is comment and we are not mid-quote — already handled)
                    boolean onlyComment = true;
                    // if multiple fields, still skip if first non-empty field starts with comment
                    // and was not quoted content from previous — simple rule: skip
                    if (onlyComment) continue;
                }
                return rec;
            }
        }

        private int read() throws IOException {
            if (nextChar != -2) {
                int c = nextChar;
                nextChar = -2;
                return c;
            }
            return in.read();
        }

        private void unread(int c) {
            nextChar = c;
        }

        private String[] readOneRecord() throws IOException {
            List<String> fields = new ArrayList<>();
            StringBuilder field = new StringBuilder();
            boolean inQuotes = false;
            boolean fieldStarted = false;
            boolean any = false;

            while (true) {
                int ci = read();
                if (ci < 0) {
                    if (!any && fields.isEmpty() && field.length() == 0) return null;
                    fields.add(field.toString());
                    return fields.toArray(new String[0]);
                }
                any = true;
                char c = (char) ci;

                if (inQuotes) {
                    if (c == quote) {
                        // doubled quote escape when escape == quote (RFC 4180)
                        int n = read();
                        if (n >= 0 && (char) n == quote && escape == quote) {
                            field.append(quote);
                        } else if (n >= 0 && escape != quote && c == escape) {
                            // shouldn't reach: escape handled below
                            unread(n);
                            field.append(c);
                        } else {
                            // end quote
                            if (n >= 0) unread(n);
                            inQuotes = false;
                        }
                    } else if (c == escape && escape != quote) {
                        int n = read();
                        if (n >= 0) field.append((char) n);
                    } else {
                        if (c == '\n') lineNumber++;
                        field.append(c);
                    }
                } else {
                    // not in quotes
                    if (!fieldStarted && comment != null && c == comment && field.length() == 0 && fields.isEmpty()) {
                        // skip rest of line
                        while (true) {
                            int n = read();
                            if (n < 0) return nextRecord(); // try next
                            if (n == '\n') { lineNumber++; return nextRecord(); }
                            if (n == '\r') {
                                int m = read();
                                if (m != '\n' && m >= 0) unread(m);
                                lineNumber++;
                                return nextRecord();
                            }
                        }
                    }
                    if (c == quote && field.length() == 0) {
                        inQuotes = true;
                        fieldStarted = true;
                    } else if (c == delim) {
                        fields.add(field.toString());
                        field.setLength(0);
                        fieldStarted = false;
                    } else if (c == '\n') {
                        lineNumber++;
                        fields.add(field.toString());
                        return fields.toArray(new String[0]);
                    } else if (c == '\r') {
                        int n = read();
                        if (n != '\n' && n >= 0) unread(n);
                        lineNumber++;
                        fields.add(field.toString());
                        return fields.toArray(new String[0]);
                    } else {
                        field.append(c);
                        fieldStarted = true;
                    }
                }
            }
        }
    }
}
