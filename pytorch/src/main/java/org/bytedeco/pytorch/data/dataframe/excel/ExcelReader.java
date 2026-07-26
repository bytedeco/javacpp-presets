package org.bytedeco.pytorch.data.dataframe.excel;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.ss.usermodel.DateUtil;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.io.IoTypeCoercion;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.*;

/**
 * Excel reader for {@code .xlsx}/{@code .xls} with type inference and
 * pandas/{@code openpyxl} interop for tabular sheets.
 */
public final class ExcelReader {
    private ExcelReader() {}

    public static DataFrame read(String path) throws Exception {
        return read(path, ExcelOptions.defaults());
    }

    public static DataFrame read(String path, ExcelOptions options) throws Exception {
        try (InputStream in = Files.newInputStream(Path.of(path))) {
            return read(in, options == null ? ExcelOptions.defaults() : options);
        }
    }

    public static DataFrame read(InputStream in, ExcelOptions options) throws Exception {
        ExcelOptions opt = options == null ? ExcelOptions.defaults() : options;
        try (Workbook wb = WorkbookFactory.create(in)) {
            Sheet sheet = resolveSheet(wb, opt);
            return readSheet(sheet, opt);
        } catch (ExcelParseException e) {
            throw e;
        } catch (Exception e) {
            throw new ExcelParseException("Failed to read Excel workbook: " + e.getMessage(), e);
        }
    }

    public static Map<String, DataFrame> readAll(String path, ExcelOptions options) throws Exception {
        ExcelOptions opt = options == null ? ExcelOptions.defaults() : options;
        try (InputStream in = Files.newInputStream(Path.of(path));
             Workbook wb = WorkbookFactory.create(in)) {
            Map<String, DataFrame> out = new LinkedHashMap<>();
            for (int i = 0; i < wb.getNumberOfSheets(); i++) {
                Sheet sheet = wb.getSheetAt(i);
                if (sheet == null) continue;
                out.put(sheet.getSheetName(), readSheet(sheet, opt));
            }
            return out;
        }
    }

    private static Sheet resolveSheet(Workbook wb, ExcelOptions opt) {
        if (opt.sheetName() != null) {
            Sheet s = wb.getSheet(opt.sheetName());
            if (s == null) {
                throw new ExcelParseException("Sheet not found: " + opt.sheetName(),
                    opt.sheetName(), -1, -1);
            }
            return s;
        }
        int idx = opt.sheetIndex();
        if (idx < 0 || idx >= wb.getNumberOfSheets()) {
            throw new ExcelParseException("Sheet index out of range: " + idx,
                null, -1, -1);
        }
        return wb.getSheetAt(idx);
    }

    private static DataFrame readSheet(Sheet sheet, ExcelOptions opt) {
        String sheetName = sheet.getSheetName();
        int firstRow = sheet.getFirstRowNum();
        int lastRow = sheet.getLastRowNum();
        if (lastRow < firstRow) return DataFrame.create();

        int rowCursor = firstRow;
        // skip leading rows
        for (int i = 0; i < opt.skipRows() && rowCursor <= lastRow; i++) rowCursor++;

        FormulaEvaluator evaluator = opt.evaluateFormulas()
            ? sheet.getWorkbook().getCreationHelper().createFormulaEvaluator()
            : null;
        DataFormatter formatter = new DataFormatter(Locale.ROOT);

        // determine column count from a scan of used range
        int maxCol = 0;
        for (int r = rowCursor; r <= lastRow; r++) {
            Row row = sheet.getRow(r);
            if (row != null) maxCol = Math.max(maxCol, (int) row.getLastCellNum());
        }
        if (maxCol <= 0) return DataFrame.create();

        String[] headers;
        if (opt.columnNames() != null && !opt.columnNames().isEmpty()) {
            headers = opt.columnNames().toArray(new String[0]);
            maxCol = Math.max(maxCol, headers.length);
        } else if (opt.header()) {
            Row headerRow = sheet.getRow(rowCursor);
            if (headerRow == null) {
                headers = defaultHeaders(maxCol);
            } else {
                headers = new String[maxCol];
                Set<String> seen = new HashSet<>();
                for (int c = 0; c < maxCol; c++) {
                    Object raw = cellValue(headerRow.getCell(c), opt, evaluator, formatter, sheetName, rowCursor, c);
                    String name = raw == null ? "" : String.valueOf(raw).trim();
                    if (name.isEmpty()) name = "col_" + c;
                    String base = name;
                    int n = 1;
                    while (!seen.add(name)) name = base + "_" + (n++);
                    headers[c] = name;
                }
            }
            rowCursor++;
        } else {
            headers = defaultHeaders(maxCol);
        }

        // collect raw object rows
        List<Object[]> rawRows = new ArrayList<>();
        int limit = opt.maxRows();
        for (int r = rowCursor; r <= lastRow; r++) {
            if (limit >= 0 && rawRows.size() >= limit) break;
            Row row = sheet.getRow(r);
            if (row == null || isEmptyRow(row, maxCol, opt, evaluator, formatter, sheetName)) {
                // skip fully empty trailing-ish rows only if nothing read yet? keep empty as null row
                // pandas keeps blank rows mid-sheet; skip completely empty
                continue;
            }
            Object[] values = new Object[headers.length];
            for (int c = 0; c < headers.length; c++) {
                Cell cell = row.getCell(c, Row.MissingCellPolicy.RETURN_BLANK_AS_NULL);
                values[c] = cellValue(cell, opt, evaluator, formatter, sheetName, r, c);
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

    private static boolean isEmptyRow(Row row, int maxCol, ExcelOptions opt,
                                      FormulaEvaluator evaluator, DataFormatter formatter,
                                      String sheetName) {
        for (int c = 0; c < maxCol; c++) {
            Object v = cellValue(row.getCell(c), opt, evaluator, formatter, sheetName, row.getRowNum(), c);
            if (v != null) return false;
        }
        return true;
    }

    private static Object cellValue(Cell cell, ExcelOptions opt, FormulaEvaluator evaluator,
                                    DataFormatter formatter, String sheet, int row, int col) {
        if (cell == null) return null;
        CellType type = cell.getCellType();
        if (type == CellType.FORMULA) {
            if (evaluator != null) {
                try {
                    type = evaluator.evaluateFormulaCell(cell);
                } catch (Exception ex) {
                    // fall back to cached
                    type = cell.getCachedFormulaResultType();
                }
            } else {
                type = cell.getCachedFormulaResultType();
            }
        }
        switch (type) {
            case BLANK:
                return null;
            case BOOLEAN:
                return cell.getBooleanCellValue();
            case STRING: {
                String s = cell.getStringCellValue();
                if (opt.isNullToken(s)) return null;
                return s;
            }
            case NUMERIC:
                if (DateUtil.isCellDateFormatted(cell)) {
                    Date d = cell.getDateCellValue();
                    if (d == null) return null;
                    LocalDateTime ldt = LocalDateTime.ofInstant(d.toInstant(), ZoneId.systemDefault());
                    // time-only vs date vs datetime: excel stores all as datetime
                    if (opt.dateAsLocalDate()) {
                        if (ldt.getHour() == 0 && ldt.getMinute() == 0 && ldt.getSecond() == 0
                            && ldt.getNano() == 0) {
                            return ldt.toLocalDate();
                        }
                        return ldt;
                    }
                    return d.toInstant();
                }
                double num = cell.getNumericCellValue();
                if (num == Math.rint(num) && !Double.isInfinite(num)
                    && num <= Long.MAX_VALUE && num >= Long.MIN_VALUE) {
                    long lv = (long) num;
                    if (lv >= Integer.MIN_VALUE && lv <= Integer.MAX_VALUE
                        && num == (int) num) {
                        // keep as long for DF INT64 inference friendliness
                        return lv;
                    }
                    return lv;
                }
                return num;
            case ERROR:
                if (opt.strict()) {
                    throw new ExcelParseException("Error cell", sheet, row, col);
                }
                return null;
            default: {
                String s = formatter.formatCellValue(cell);
                if (opt.isNullToken(s)) return null;
                return s;
            }
        }
    }

    private static String[] defaultHeaders(int n) {
        String[] h = new String[n];
        for (int i = 0; i < n; i++) h[i] = "col_" + i;
        return h;
    }
}
