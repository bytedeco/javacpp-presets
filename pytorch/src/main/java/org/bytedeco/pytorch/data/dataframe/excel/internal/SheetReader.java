package org.bytedeco.pytorch.data.dataframe.excel.internal;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamReader;
import java.io.InputStream;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Streaming sheet reader for xl/worksheets/sheetN.xml.
 * Returns rows as Object[] (nulls for blanks); column count expands as needed.
 */
public final class SheetReader {
    public static final class SheetData {
        public final List<Object[]> rows;
        public final int maxCol;

        public SheetData(List<Object[]> rows, int maxCol) {
            this.rows = rows;
            this.maxCol = maxCol;
        }
    }

    private SheetReader() {}

    public static SheetData read(InputStream in, SharedStringsTable sst, StylesTable styles,
                                 boolean dateAsLocalDate) throws Exception {
        List<Object[]> rows = new ArrayList<>();
        int maxCol = 0;
        if (in == null) return new SheetData(rows, 0);

        XMLInputFactory factory = XMLInputFactory.newFactory();
        factory.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, false);
        factory.setProperty(XMLInputFactory.SUPPORT_DTD, false);
        XMLStreamReader r = factory.createXMLStreamReader(in, "UTF-8");

        int currentRowIdx = -1;
        List<Object> currentCells = null;
        String cellRef = null;
        String cellType = null;
        int cellStyle = -1;
        boolean inV = false;
        boolean inT = false;
        boolean inIs = false;
        StringBuilder text = new StringBuilder();

        try {
            while (r.hasNext()) {
                int ev = r.next();
                if (ev == XMLStreamConstants.START_ELEMENT) {
                    String local = r.getLocalName();
                    if ("row".equals(local)) {
                        int rIdx = intAttr(r, "r", rows.size() + 1) - 1;
                        while (rows.size() < rIdx) rows.add(new Object[0]);
                        currentRowIdx = rIdx;
                        currentCells = new ArrayList<>();
                    } else if ("c".equals(local)) {
                        cellRef = attr(r, "r");
                        cellType = attr(r, "t");
                        cellStyle = intAttr(r, "s", -1);
                        text.setLength(0);
                    } else if ("v".equals(local)) {
                        inV = true;
                        text.setLength(0);
                    } else if ("t".equals(local) && (inIs || "inlineStr".equals(cellType))) {
                        inT = true;
                        // keep accumulating if multiple t nodes
                    } else if ("is".equals(local)) {
                        inIs = true;
                        text.setLength(0);
                    }
                } else if (ev == XMLStreamConstants.CHARACTERS || ev == XMLStreamConstants.CDATA) {
                    if (inV || inT) text.append(r.getText());
                } else if (ev == XMLStreamConstants.END_ELEMENT) {
                    String local = r.getLocalName();
                    if ("v".equals(local)) {
                        inV = false;
                        putCell(currentCells, cellRef, cellType, cellStyle, text.toString(),
                            sst, styles, dateAsLocalDate);
                        maxCol = Math.max(maxCol, colIndex(cellRef) + 1);
                    } else if ("t".equals(local)) {
                        inT = false;
                    } else if ("is".equals(local)) {
                        inIs = false;
                        putCell(currentCells, cellRef, "inlineStr", cellStyle, text.toString(),
                            sst, styles, dateAsLocalDate);
                        maxCol = Math.max(maxCol, colIndex(cellRef) + 1);
                    } else if ("c".equals(local)) {
                        cellRef = null;
                        cellType = null;
                        cellStyle = -1;
                    } else if ("row".equals(local)) {
                        if (currentCells != null) {
                            while (rows.size() <= currentRowIdx) rows.add(new Object[0]);
                            Object[] arr = currentCells.toArray();
                            rows.set(currentRowIdx, arr);
                            maxCol = Math.max(maxCol, arr.length);
                        }
                        currentCells = null;
                        currentRowIdx = -1;
                    }
                }
            }
        } finally {
            r.close();
        }
        return new SheetData(rows, maxCol);
    }

    private static void putCell(List<Object> cells, String ref, String type, int style,
                                String raw, SharedStringsTable sst, StylesTable styles,
                                boolean dateAsLocalDate) {
        if (cells == null || ref == null) return;
        int col = colIndex(ref);
        while (cells.size() <= col) cells.add(null);
        cells.set(col, decode(type, style, raw, sst, styles, dateAsLocalDate));
    }

    private static Object decode(String type, int style, String raw,
                                 SharedStringsTable sst, StylesTable styles,
                                 boolean dateAsLocalDate) {
        if (raw == null) return null;
        if (type == null || type.isEmpty() || "n".equals(type)) {
            if (raw.isEmpty()) return null;
            double num;
            try { num = Double.parseDouble(raw); }
            catch (NumberFormatException e) { return raw; }
            if (style >= 0 && styles != null && styles.isDateStyle(style)) {
                LocalDateTime ldt = ExcelDateUtil.fromSerial(num);
                if (dateAsLocalDate) {
                    if (ldt.toLocalTime().equals(java.time.LocalTime.MIDNIGHT)) {
                        return ldt.toLocalDate();
                    }
                    return ldt;
                }
                return ldt.atZone(java.time.ZoneId.systemDefault()).toInstant();
            }
            if (num == Math.rint(num) && !Double.isInfinite(num)
                && num <= Long.MAX_VALUE && num >= Long.MIN_VALUE) {
                return (long) num;
            }
            return num;
        }
        switch (type) {
            case "s": {
                int idx;
                try { idx = Integer.parseInt(raw.trim()); }
                catch (NumberFormatException e) { return raw; }
                return sst == null ? raw : sst.get(idx);
            }
            case "inlineStr":
            case "str":
                return raw;
            case "b":
                return "1".equals(raw) || "true".equalsIgnoreCase(raw);
            case "e":
                return null;
            default:
                return raw;
        }
    }

    /** A1 → 0-based column index. */
    public static int colIndex(String cellRef) {
        if (cellRef == null || cellRef.isEmpty()) return 0;
        int col = 0;
        for (int i = 0; i < cellRef.length(); i++) {
            char c = cellRef.charAt(i);
            if (c >= 'A' && c <= 'Z') col = col * 26 + (c - 'A' + 1);
            else if (c >= 'a' && c <= 'z') col = col * 26 + (c - 'a' + 1);
            else break;
        }
        return Math.max(0, col - 1);
    }

    public static String colName(int index) {
        StringBuilder sb = new StringBuilder();
        int n = index + 1;
        while (n > 0) {
            int rem = (n - 1) % 26;
            sb.insert(0, (char) ('A' + rem));
            n = (n - 1) / 26;
        }
        return sb.toString();
    }

    private static String attr(XMLStreamReader r, String name) {
        String v = r.getAttributeValue(null, name);
        if (v != null) return v;
        for (int i = 0; i < r.getAttributeCount(); i++) {
            if (name.equals(r.getAttributeLocalName(i))) return r.getAttributeValue(i);
        }
        return null;
    }

    private static int intAttr(XMLStreamReader r, String name, int def) {
        String v = attr(r, name);
        if (v == null || v.isEmpty()) return def;
        try { return Integer.parseInt(v); } catch (NumberFormatException e) { return def; }
    }
}
