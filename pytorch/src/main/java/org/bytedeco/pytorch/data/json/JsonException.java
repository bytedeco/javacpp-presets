package org.bytedeco.pytorch.data.json;

/**
 * JSON parse / write error with optional line/column location (1-based).
 */
public class JsonException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    private final long line;
    private final long column;
    private final long offset;

    public JsonException(String message) {
        this(message, -1, -1, -1, null);
    }

    public JsonException(String message, Throwable cause) {
        this(message, -1, -1, -1, cause);
    }

    public JsonException(String message, long line, long column, long offset) {
        this(message, line, column, offset, null);
    }

    public JsonException(String message, long line, long column, long offset, Throwable cause) {
        super(format(message, line, column, offset), cause);
        this.line = line;
        this.column = column;
        this.offset = offset;
    }

    public long line() { return line; }
    public long column() { return column; }
    public long offset() { return offset; }

    private static String format(String message, long line, long column, long offset) {
        if (line < 0 && column < 0) return message;
        StringBuilder sb = new StringBuilder(message == null ? "JSON error" : message);
        sb.append(" (");
        if (line >= 0) sb.append("line ").append(line);
        if (column >= 0) {
            if (line >= 0) sb.append(", ");
            sb.append("col ").append(column);
        }
        if (offset >= 0) sb.append(", offset ").append(offset);
        sb.append(')');
        return sb.toString();
    }
}
