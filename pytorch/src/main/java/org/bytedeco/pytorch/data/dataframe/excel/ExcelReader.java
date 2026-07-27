package org.bytedeco.pytorch.data.dataframe.excel;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.excel.internal.Biff8Workbook;
import org.bytedeco.pytorch.data.dataframe.excel.internal.OoxmlZip;
import org.bytedeco.pytorch.data.dataframe.excel.internal.SheetReader;
import org.bytedeco.pytorch.data.dataframe.io.IoTypeCoercion;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Excel reader for {@code .xlsx} (pure OOXML) and best-effort {@code .xls} (BIFF8).
 * No Apache POI dependency.
 *
 * <p>Formula cells expose cached values only (no formula engine).
 */
public final class ExcelReader {
    private ExcelReader() {}

    public static DataFrame read(String path) throws Exception {
        return read(path, ExcelOptions.defaults());
    }

    public static DataFrame read(String path, ExcelOptions options) throws Exception {
        try (InputStream in = Files.newInputStream(Path.of(path))) {
            return read(in, options == null ? ExcelOptions.defaults() : options, path);
        }
    }

    public static DataFrame read(InputStream in, ExcelOptions options) throws Exception {
        return read(in, options, null);
    }

    private static DataFrame read(InputStream in, ExcelOptions options, String pathHint) throws Exception {
        ExcelOptions opt = options == null ? ExcelOptions.defaults() : options;
        BufferedInputStream bin = in instanceof BufferedInputStream
            ? (BufferedInputStream) in : new BufferedInputStream(in);
        bin.mark(8);
        byte[] magic = new byte[8];
        int n = bin.read(magic);
        bin.reset();
        try {
            if (n >= 2 && magic[0] == 'P' && magic[1] == 'K') {
                // OOXML zip
                OoxmlZip.Package pkg = OoxmlZip.open(bin);
                OoxmlZip.SheetMeta sheet = resolveSheet(pkg, opt);
                return readOoxmlSheet(pkg, sheet, opt);
            }
            if (n >= 4 && (magic[0] & 0xFF) == 0xD0 && (magic[1] & 0xFF) == 0xCF
                && (magic[2] & 0xFF) == 0x11 && (magic[3] & 0xFF) == 0xE0) {
                return readXls(bin, opt);
            }
            // path hint extension
            if (pathHint != null) {
                String lower = pathHint.toLowerCase();
                if (lower.endsWith(".xlsx") || lower.endsWith(".xlsm")) {
                    OoxmlZip.Package pkg = OoxmlZip.open(bin);
                    return readOoxmlSheet(pkg, resolveSheet(pkg, opt), opt);
                }
                if (lower.endsWith(".xls")) {
                    return readXls(bin, opt);
                }
            }
            throw new ExcelParseException("Unrecognized Excel format (expected .xlsx zip or .xls OLE)");
        } catch (ExcelParseException e) {
            throw e;
        } catch (Exception e) {
            throw new ExcelParseException("Failed to read Excel workbook: " + e.getMessage(), e);
        }
    }

    public static Map<String, DataFrame> readAll(String path, ExcelOptions options) throws Exception {
        ExcelOptions opt = options == null ? ExcelOptions.defaults() : options;
        try (InputStream in = new BufferedInputStream(Files.newInputStream(Path.of(path)))) {
            in.mark(8);
            byte[] magic = new byte[8];
            int n = in.read(magic);
            in.reset();
            Map<String, DataFrame> out = new LinkedHashMap<>();
            if (n >= 2 && magic[0] == 'P' && magic[1] == 'K') {
                OoxmlZip.Package pkg = OoxmlZip.open(in);
                for (OoxmlZip.SheetMeta sm : pkg.sheets) {
                    out.put(sm.name, readOoxmlSheet(pkg, sm, opt));
                }
                return out;
            }
            if (n >= 4 && (magic[0] & 0xFF) == 0xD0) {
                Biff8Workbook wb = Biff8Workbook.read(in);
                for (Biff8Workbook.Sheet s : wb.sheets()) {
                    out.put(s.name, rowsToDataFrame(s.rows, s.maxCol, opt, s.name));
                }
                return out;
            }
            throw new ExcelParseException("Unrecognized Excel format for readAll");
        }
    }

    private static OoxmlZip.SheetMeta resolveSheet(OoxmlZip.Package pkg, ExcelOptions opt) {
        if (pkg.sheets.isEmpty()) {
            throw new ExcelParseException("Workbook has no sheets");
        }
        if (opt.sheetName() != null) {
            for (OoxmlZip.SheetMeta sm : pkg.sheets) {
                if (opt.sheetName().equals(sm.name)) return sm;
            }
            throw new ExcelParseException("Sheet not found: " + opt.sheetName(),
                opt.sheetName(), -1, -1);
        }
        int idx = opt.sheetIndex();
        if (idx < 0 || idx >= pkg.sheets.size()) {
            throw new ExcelParseException("Sheet index out of range: " + idx, null, -1, -1);
        }
        return pkg.sheets.get(idx);
    }

    private static DataFrame readOoxmlSheet(OoxmlZip.Package pkg, OoxmlZip.SheetMeta sheet,
                                           ExcelOptions opt) throws Exception {
        byte[] part = pkg.parts.get(sheet.path);
        if (part == null) {
            // try alternate path forms
            for (String k : pkg.parts.keySet()) {
                if (k.endsWith(sheet.path) || k.endsWith("/" + sheet.path.replace("xl/", ""))) {
                    part = pkg.parts.get(k);
                    break;
                }
            }
        }
        if (part == null) {
            throw new ExcelParseException("Sheet part missing: " + sheet.path, sheet.name, -1, -1);
        }
        SheetReader.SheetData data = SheetReader.read(
            new ByteArrayInputStream(part), pkg.sst, pkg.styles, opt.dateAsLocalDate());
        return rowsToDataFrame(data.rows, data.maxCol, opt, sheet.name);
    }

    private static DataFrame readXls(InputStream in, ExcelOptions opt) throws Exception {
        Biff8Workbook wb = Biff8Workbook.read(in);
        if (wb.sheets().isEmpty()) return DataFrame.create();
        Biff8Workbook.Sheet sheet;
        if (opt.sheetName() != null) {
            sheet = null;
            for (Biff8Workbook.Sheet s : wb.sheets()) {
                if (opt.sheetName().equals(s.name)) { sheet = s; break; }
            }
            if (sheet == null) {
                throw new ExcelParseException("Sheet not found: " + opt.sheetName(),
                    opt.sheetName(), -1, -1);
            }
        } else {
            int idx = opt.sheetIndex();
            if (idx < 0 || idx >= wb.sheets().size()) {
                throw new ExcelParseException("Sheet index out of range: " + idx);
            }
            sheet = wb.sheets().get(idx);
        }
        return rowsToDataFrame(sheet.rows, sheet.maxCol, opt, sheet.name);
    }

    private static DataFrame rowsToDataFrame(List<Object[]> allRows, int maxCol,
                                             ExcelOptions opt, String sheetName) {
        if (allRows == null || allRows.isEmpty() || maxCol <= 0) {
            // still allow explicit column names
            if (opt.columnNames() != null && !opt.columnNames().isEmpty()) {
                DataFrame df = DataFrame.create();
                for (String n : opt.columnNames()) df.addColumn(n, Column.DType.STRING);
                return df;
            }
            return DataFrame.create();
        }

        int rowCursor = 0;
        for (int i = 0; i < opt.skipRows() && rowCursor < allRows.size(); i++) rowCursor++;

        String[] headers;
        if (opt.columnNames() != null && !opt.columnNames().isEmpty()) {
            headers = opt.columnNames().toArray(new String[0]);
            maxCol = Math.max(maxCol, headers.length);
        } else if (opt.header()) {
            if (rowCursor >= allRows.size()) {
                headers = defaultHeaders(maxCol);
            } else {
                Object[] headerRow = allRows.get(rowCursor);
                headers = new String[maxCol];
                Set<String> seen = new HashSet<>();
                for (int c = 0; c < maxCol; c++) {
                    Object raw = c < headerRow.length ? headerRow[c] : null;
                    String name = raw == null ? "" : String.valueOf(raw).trim();
                    if (name.isEmpty()) name = "col_" + c;
                    if (opt.isNullToken(name)) name = "col_" + c;
                    String base = name;
                    int n = 1;
                    while (!seen.add(name)) name = base + "_" + (n++);
                    headers[c] = name;
                }
                rowCursor++;
            }
        } else {
            headers = defaultHeaders(maxCol);
        }

        List<Object[]> rawRows = new ArrayList<>();
        int limit = opt.maxRows();
        for (int r = rowCursor; r < allRows.size(); r++) {
            if (limit >= 0 && rawRows.size() >= limit) break;
            Object[] row = allRows.get(r);
            if (isEmptyRow(row, headers.length, opt)) continue;
            Object[] values = new Object[headers.length];
            for (int c = 0; c < headers.length; c++) {
                Object v = (row != null && c < row.length) ? row[c] : null;
                if (v instanceof String && opt.isNullToken((String) v)) v = null;
                values[c] = v;
            }
            rawRows.add(values);
        }

        Column.DType[] dtypes = resolveDtypes(headers, rawRows, opt);
        DataFrame df = DataFrame.create();
        for (int i = 0; i < headers.length; i++) df.addColumn(headers[i], dtypes[i]);
        for (Object[] row : rawRows) {
            int ri = df.addEmptyRow();
            for (int c = 0; c < headers.length; c++) {
                Object v = c < row.length ? row[c] : null;
                try {
                    df.set(ri, headers[c], v == null ? null : IoTypeCoercion.coerce(v, dtypes[c]));
                } catch (Exception ex) {
                    if (opt.strict()) {
                        throw new ExcelParseException("Cannot coerce value to " + dtypes[c] + ": " + v,
                            sheetName, ri, c, ex);
                    }
                    df.set(ri, headers[c], v == null ? null : String.valueOf(v));
                }
            }
        }
        return df;
    }

    private static Column.DType[] resolveDtypes(String[] headers, List<Object[]> rows, ExcelOptions opt) {
        Column.DType[] dtypes = new Column.DType[headers.length];
        if (opt.schema() != null) {
            for (int i = 0; i < headers.length; i++) {
                Column.DType t = opt.schema().get(headers[i]);
                dtypes[i] = t != null ? t : Column.DType.STRING;
            }
            return dtypes;
        }
        if (!opt.inferSchema()) {
            Arrays.fill(dtypes, Column.DType.STRING);
            return dtypes;
        }
        int sample = Math.min(opt.inferSampleSize(), rows.size());
        for (int c = 0; c < headers.length; c++) {
            Column.DType acc = null;
            for (int r = 0; r < sample; r++) {
                Object v = rows.get(r)[c];
                if (v == null) continue;
                Column.DType t = IoTypeCoercion.inferFromObject(v);
                acc = acc == null ? t : IoTypeCoercion.widen(acc, t);
            }
            dtypes[c] = acc == null ? Column.DType.STRING : acc;
        }
        return dtypes;
    }

    private static boolean isEmptyRow(Object[] row, int maxCol, ExcelOptions opt) {
        if (row == null) return true;
        for (int c = 0; c < maxCol && c < row.length; c++) {
            Object v = row[c];
            if (v == null) continue;
            if (v instanceof String && opt.isNullToken((String) v)) continue;
            if (v instanceof String && ((String) v).isEmpty()) continue;
            return false;
        }
        return true;
    }

    private static String[] defaultHeaders(int n) {
        String[] h = new String[Math.max(0, n)];
        for (int i = 0; i < h.length; i++) h[i] = "col_" + i;
        return h;
    }
}
