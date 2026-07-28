package org.bytedeco.pytorch.dataframe.milvus;

/**
 * Unchecked failure from {@link Milvus} REST / SPI operations.
 */
public class MilvusException extends RuntimeException {
    private final int status;
    private final String operation;

    public MilvusException(String message) {
        this(message, null, -1, null);
    }

    public MilvusException(String message, Throwable cause) {
        this(message, cause, -1, null);
    }

    public MilvusException(String message, int status, String operation) {
        this(message, null, status, operation);
    }

    public MilvusException(String message, Throwable cause, int status, String operation) {
        super(message, cause);
        this.status = status;
        this.operation = operation;
    }

    /** HTTP status if known, else {@code -1}. */
    public int status() { return status; }

    /** Logical operation name (e.g. {@code collections/create}). */
    public String operation() { return operation; }
}
