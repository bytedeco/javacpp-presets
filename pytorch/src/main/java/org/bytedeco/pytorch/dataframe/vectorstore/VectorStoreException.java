package org.bytedeco.pytorch.dataframe.vectorstore;

/**
 * Unchecked failure from a {@link VectorStore} operation (HTTP error, RESP error,
 * missing driver, schema mismatch, …).
 */
public class VectorStoreException extends RuntimeException {
    private final int status;
    private final String backend;

    public VectorStoreException(String message) {
        this(message, null, -1, null);
    }

    public VectorStoreException(String message, Throwable cause) {
        this(message, cause, -1, null);
    }

    public VectorStoreException(String message, int status, String backend) {
        this(message, null, status, backend);
    }

    public VectorStoreException(String message, Throwable cause, int status, String backend) {
        super(message, cause);
        this.status = status;
        this.backend = backend;
    }

    /** HTTP / protocol status if known, else {@code -1}. */
    public int status() { return status; }

    /** Backend label ({@code "qdrant"}, {@code "redis"}, …) if known. */
    public String backend() { return backend; }
}
