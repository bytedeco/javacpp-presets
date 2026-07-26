package org.bytedeco.pytorch.data.dataframe.excel;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.streaming.SXSSFWorkbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.Date;
import java.util.Map;

/**
 * Excel writer ({@code .xlsx}) with dtype-aware cells. Uses SXSSF for larger frames.
 */
public final class ExcelWriter {
    private ExcelWriter() {}

    public static void write(DataFrame df, String path) throws Exception {
        write(df, path, ExcelOptions.defaults());
    }

    public static void write(DataFrame df, String path, ExcelOptions options) throws Exception {
        ExcelOptions opt = options == null ? ExcelOptions.defaults() : options;
        try (OutputStream out = Files.newOutputStream(Path.of(path))) {
            write(df, out, opt);
        }
    }

    public static void write(DataFrame df, OutputStream out, ExcelOptions options) throws Exception {
        ExcelOptions opt = options == null ? ExcelOptions.defaults() : options;
        boolean large = df != null && df.rowCount() > 5_000;
        Workbook wb = large ? new SXSSFWorkbook(500) : new XSSFWorkbook();
        try {
            writeSheet(wb, opt.writeSheetName(), df, opt);
            wb.write(out);
        } finally {
            if (wb instanceof SXSSFWorkbook) {
                ((SXSSFWorkbook) wb).dispose();
            }
            wb.close();
        }
    }

    public static void writeSheets(String path, Map<String, DataFrame> sheets, ExcelOptions options)
            throws Exception {
        ExcelOptions opt = options == null ? ExcelOptions.defaults() : options;
        try (Workbook wb = new XSSFWorkbook();
             OutputStream out = Files.newOutputStream(Path.of(path))) {
            if (sheets == null || sheets.isEmpty()) {
                writeSheet(wb, "Sheet1", DataFrame.create(), opt);
            } else {
                for (Map.Entry<String, DataFrame> e : sheets.entrySet()) {
                    String name = sanitizeSheetName(e.getKey());
                    writeSheet(wb, name, e.getValue() == null ? DataFrame.create() : e.getValue(), opt);
                }
            }
            wb.write(out);
        }
    }

    private static void writeSheet(Workbook wb, String sheetName, DataFrame df, ExcelOptions opt) {
        Sheet sheet = wb.createSheet(sanitizeSheetName(sheetName));
        CreationHelper helper = wb.getCreationHelper();
        CellStyle dateStyle = wb.createCellStyle();
        dateStyle.setDataFormat(helper.createDataFormat().getFormat("yyyy-mm-dd"));
        CellStyle dateTimeStyle = wb.createCellStyle();
        dateTimeStyle.setDataFormat(helper.createDataFormat().getFormat("yyyy-mm-dd hh:mm:ss"));

        int rowIdx = 0;
        int cols = df.columnCount();
        if (opt.header()) {
            Row header = sheet.createRow(rowIdx++);
            for (int c = 0; c < cols; c++) {
                Cell cell = header.createCell(c, CellType.STRING);
                cell.setCellValue(df.column(c).name());
            }
            if (opt.freezeHeader()) {
                sheet.createFreezePane(0, 1);
            }
        }

        for (int r = 0; r < df.rowCount(); r++) {
            Row row = sheet.createRow(rowIdx++);
            for (int c = 0; c < cols; c++) {
                Column col = df.column(c);
                Object val = col.get(r);
                Cell cell = row.createCell(c);
                setCell(cell, val, col.dtype(), opt, dateStyle, dateTimeStyle);
            }
        }
    }

    private static void setCell(Cell cell, Object val, Column.DType dtype,
                                ExcelOptions opt, CellStyle dateStyle, CellStyle dateTimeStyle) {
        if (val == null) {
            if (opt.writeNullToken() != null && !opt.writeNullToken().isEmpty()) {
                cell.setCellValue(opt.writeNullToken());
            } else {
                cell.setBlank();
            }
            return;
        }
        try {
            switch (dtype) {
                case INT32:
                case INT64:
                case FLOAT32:
                case FLOAT64:
                    if (val instanceof Number) {
                        cell.setCellValue(((Number) val).doubleValue());
                    } else {
                        cell.setCellValue(Double.parseDouble(String.valueOf(val)));
                    }
                    break;
                case BOOLEAN:
                    if (val instanceof Boolean) cell.setCellValue((Boolean) val);
                    else cell.setCellValue(Boolean.parseBoolean(String.valueOf(val)));
                    break;
                case DATE:
                    if (val instanceof LocalDate) {
                        cell.setCellValue(java.sql.Date.valueOf((LocalDate) val));
                        cell.setCellStyle(dateStyle);
                    } else if (val instanceof Date) {
                        cell.setCellValue((Date) val);
                        cell.setCellStyle(dateStyle);
                    } else {
                        cell.setCellValue(String.valueOf(val));
                    }
                    break;
                case DATETIME:
                    if (val instanceof LocalDateTime) {
                        Date d = Date.from(((LocalDateTime) val).atZone(ZoneId.systemDefault()).toInstant());
                        cell.setCellValue(d);
                        cell.setCellStyle(dateTimeStyle);
                    } else if (val instanceof Instant) {
                        cell.setCellValue(Date.from((Instant) val));
                        cell.setCellStyle(dateTimeStyle);
                    } else if (val instanceof Date) {
                        cell.setCellValue((Date) val);
                        cell.setCellStyle(dateTimeStyle);
                    } else {
                        cell.setCellValue(String.valueOf(val));
                    }
                    break;
                case TIME:
                    if (val instanceof LocalTime) {
                        cell.setCellValue(val.toString());
                    } else {
                        cell.setCellValue(String.valueOf(val));
                    }
                    break;
                default:
                    cell.setCellValue(String.valueOf(val));
            }
        } catch (Exception ex) {
            cell.setCellValue(String.valueOf(val));
        }
    }

    private static String sanitizeSheetName(String name) {
        if (name == null || name.isEmpty()) return "Sheet1";
        // Excel sheet name limits: 31 chars, no \ / ? * [ ]
        String s = name.replaceAll("[\\\\/?*\\[\\]]", "_");
        if (s.length() > 31) s = s.substring(0, 31);
        if (s.isEmpty()) s = "Sheet1";
        return s;
    }
}
