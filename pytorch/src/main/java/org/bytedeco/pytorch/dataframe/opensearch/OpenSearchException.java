package org.bytedeco.pytorch.dataframe.opensearch;

/**
 * Unchecked failure from {@link OpenSearch} REST / SPI operations.
 */
public class OpenSearchException extends RuntimeException {
    private final int status;
    private final String operation;

    public OpenSearchException(String message) {
        this(message, null, -1, null);
    }

    public OpenSearchException(String message, Throwable cause) {
        this(message, cause, -1, null);
    }

    public OpenSearchException(String message, int status, String operation) {
        this(message, null, status, operation);
    }

    public OpenSearchException(String message, Throwable cause, int status, String operation) {
        super(message, cause);
        this.status = status;
        this.operation = operation;
    }

    public int status() { return status; }
    public String operation() { return operation; }
}
