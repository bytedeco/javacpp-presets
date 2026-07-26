package org.bytedeco.pytorch.data.dataframe.excel;

/**
 * Excel parse/write error with optional sheet/row/column context.
 */
public class ExcelParseException extends RuntimeException {
    private final String sheet;
    private final int row;
    private final int col;

    public ExcelParseException(String message) {
        this(message, null, -1, -1, null);
    }

    public ExcelParseException(String message, Throwable cause) {
        this(message, null, -1, -1, cause);
    }

    public ExcelParseException(String message, String sheet, int row, int col) {
        this(message, sheet, row, col, null);
    }

    public ExcelParseException(String message, String sheet, int row, int col, Throwable cause) {
        super(format(message, sheet, row, col), cause);
        this.sheet = sheet;
        this.row = row;
        this.col = col;
    }

    public String sheet() { return sheet; }
    public int row() { return row; }
    public int col() { return col; }

    private static String format(String message, String sheet, int row, int col) {
        StringBuilder sb = new StringBuilder(message == null ? "Excel error" : message);
        if (sheet != null || row >= 0 || col >= 0) {
            sb.append(" [");
            if (sheet != null) sb.append("sheet=").append(sheet);
            if (row >= 0) {
                if (sheet != null) sb.append(", ");
                sb.append("row=").append(row);
            }
            if (col >= 0) {
                if (sheet != null || row >= 0) sb.append(", ");
                sb.append("col=").append(col);
            }
            sb.append(']');
        }
        return sb.toString();
    }
}
