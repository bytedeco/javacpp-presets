package org.bytedeco.pytorch.dataframe.csv;

/**
 * Exception thrown when CSV parsing fails in strict mode or encounters unrecoverable errors.
 */
public final class CsvParseException extends RuntimeException {
    private final long lineNumber;
    private final int fieldIndex;
    private final String fieldPreview;

    public CsvParseException(String message) {
        this(message, -1, -1, null, null);
    }

    public CsvParseException(String message, long lineNumber, int fieldIndex, String fieldPreview) {
        this(message, lineNumber, fieldIndex, fieldPreview, null);
    }

    public CsvParseException(String message, long lineNumber, int fieldIndex, String fieldPreview, Throwable cause) {
        super(format(message, lineNumber, fieldIndex, fieldPreview), cause);
        this.lineNumber = lineNumber;
        this.fieldIndex = fieldIndex;
        this.fieldPreview = fieldPreview;
    }

    public long lineNumber() { return lineNumber; }
    public int fieldIndex() { return fieldIndex; }
    public String fieldPreview() { return fieldPreview; }

    private static String format(String message, long line, int field, String preview) {
        StringBuilder sb = new StringBuilder(message == null ? "CSV parse error" : message);
        if (line >= 0) sb.append(" [line=").append(line).append(']');
        if (field >= 0) sb.append(" [field=").append(field).append(']');
        if (preview != null && !preview.isEmpty()) {
            String p = preview.length() > 64 ? preview.substring(0, 64) + "..." : preview;
            sb.append(" [value=\"").append(p).append("\"]");
        }
        return sb.toString();
    }
}
