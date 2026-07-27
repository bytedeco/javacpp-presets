package org.bytedeco.pytorch.data.dataframe.excel.internal;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec;

import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.Date;

/**
 * Writes a single worksheet XML part for a DataFrame.
 */
public final class SheetWriter {
    private SheetWriter() {}

    public static void write(OutputStream out, DataFrame df, SharedStringsTable sst,
                             boolean header, boolean freezeHeader, String writeNullToken) throws Exception {
        Writer w = new OutputStreamWriter(out, StandardCharsets.UTF_8);
        w.write("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>");
        w.write("<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"");
        w.write(" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">");
        if (freezeHeader && header) {
            w.write("<sheetViews><sheetView workbookViewId=\"0\">");
            w.write("<pane ySplit=\"1\" topLeftCell=\"A2\" activePane=\"bottomLeft\" state=\"frozen\"/>");
            w.write("</sheetView></sheetViews>");
        }
        w.write("<sheetData>");

        int rowNum = 1; // 1-based
        int cols = df == null ? 0 : df.columnCount();
        if (header && df != null) {
            w.write("<row r=\"");
            w.write(Integer.toString(rowNum));
            w.write("\">");
            for (int c = 0; c < cols; c++) {
                String name = df.column(c).name();
                int si = sst.add(name == null ? "" : name);
                writeSharedStringCell(w, SheetReader.colName(c) + rowNum, si);
            }
            w.write("</row>");
            rowNum++;
        }

        if (df != null) {
            for (int r = 0; r < df.rowCount(); r++) {
                w.write("<row r=\"");
                w.write(Integer.toString(rowNum));
                w.write("\">");
                for (int c = 0; c < cols; c++) {
                    Column col = df.column(c);
                    Object val = col.get(r);
                    String ref = SheetReader.colName(c) + rowNum;
                    writeValue(w, ref, val, col.dtype(), sst, writeNullToken);
                }
                w.write("</row>");
                rowNum++;
            }
        }
        w.write("</sheetData></worksheet>");
        w.flush();
    }

    private static void writeValue(Writer w, String ref, Object val, Column.DType dtype,
                                   SharedStringsTable sst, String writeNullToken) throws Exception {
        if (val == null) {
            if (writeNullToken != null && !writeNullToken.isEmpty()) {
                int si = sst.add(writeNullToken);
                writeSharedStringCell(w, ref, si);
            }
            // blank cell: omit
            return;
        }
        try {
            switch (dtype) {
                case INT32:
                case INT64:
                case FLOAT32:
                case FLOAT64: {
                    double d = val instanceof Number ? ((Number) val).doubleValue()
                        : Double.parseDouble(String.valueOf(val));
                    writeNumberCell(w, ref, d, StylesTable.STYLE_GENERAL);
                    break;
                }
                case BOOLEAN: {
                    boolean b = val instanceof Boolean ? (Boolean) val
                        : Boolean.parseBoolean(String.valueOf(val));
                    writeBoolCell(w, ref, b);
                    break;
                }
                case DATE: {
                    double serial;
                    if (val instanceof LocalDate) serial = ExcelDateUtil.toSerial((LocalDate) val);
                    else if (val instanceof Date) serial = ExcelDateUtil.toSerial((Date) val);
                    else if (val instanceof LocalDateTime) serial = ExcelDateUtil.toSerial((LocalDateTime) val);
                    else {
                        writeStringCell(w, ref, String.valueOf(val), sst);
                        break;
                    }
                    writeNumberCell(w, ref, serial, StylesTable.STYLE_DATE);
                    break;
                }
                case DATETIME: {
                    double serial;
                    if (val instanceof LocalDateTime) serial = ExcelDateUtil.toSerial((LocalDateTime) val);
                    else if (val instanceof Instant)
                        serial = ExcelDateUtil.toSerial((Instant) val, ZoneId.systemDefault());
                    else if (val instanceof Date) serial = ExcelDateUtil.toSerial((Date) val);
                    else if (val instanceof LocalDate) serial = ExcelDateUtil.toSerial((LocalDate) val);
                    else {
                        writeStringCell(w, ref, String.valueOf(val), sst);
                        break;
                    }
                    writeNumberCell(w, ref, serial, StylesTable.STYLE_DATETIME);
                    break;
                }
                case TIME: {
                    if (val instanceof LocalTime) writeStringCell(w, ref, val.toString(), sst);
                    else writeStringCell(w, ref, String.valueOf(val), sst);
                    break;
                }
                case VECTOR:
                case EMBEDDING:
                case LIST:
                case MAP:
                case STRUCT:
                case JSON:
                case BINARY:
                case TENSOR:
                    writeStringCell(w, ref, ComplexCellCodec.encodeText(val), sst);
                    break;
                default:
                    // Nested Java values without typed dtype still JSON-encode
                    if (val instanceof java.util.Map || val instanceof java.util.List
                        || val instanceof float[] || val instanceof double[]
                        || val instanceof int[] || val instanceof long[]
                        || val instanceof boolean[]) {
                        writeStringCell(w, ref, ComplexCellCodec.encodeText(val), sst);
                    } else {
                        writeStringCell(w, ref, String.valueOf(val), sst);
                    }
            }
        } catch (Exception ex) {
            writeStringCell(w, ref, String.valueOf(val), sst);
        }
    }

    private static void writeNumberCell(Writer w, String ref, double v, int style) throws Exception {
        w.write("<c r=\"");
        w.write(ref);
        if (style != StylesTable.STYLE_GENERAL) {
            w.write("\" s=\"");
            w.write(Integer.toString(style));
        }
        w.write("\"><v>");
        if (v == Math.rint(v) && !Double.isInfinite(v) && Math.abs(v) < 1e15) {
            w.write(Long.toString((long) v));
        } else {
            w.write(Double.toString(v));
        }
        w.write("</v></c>");
    }

    private static void writeBoolCell(Writer w, String ref, boolean b) throws Exception {
        w.write("<c r=\"");
        w.write(ref);
        w.write("\" t=\"b\"><v>");
        w.write(b ? "1" : "0");
        w.write("</v></c>");
    }

    private static void writeStringCell(Writer w, String ref, String s, SharedStringsTable sst) throws Exception {
        int si = sst.add(s == null ? "" : s);
        writeSharedStringCell(w, ref, si);
    }

    private static void writeSharedStringCell(Writer w, String ref, int si) throws Exception {
        w.write("<c r=\"");
        w.write(ref);
        w.write("\" t=\"s\"><v>");
        w.write(Integer.toString(si));
        w.write("</v></c>");
    }
}
