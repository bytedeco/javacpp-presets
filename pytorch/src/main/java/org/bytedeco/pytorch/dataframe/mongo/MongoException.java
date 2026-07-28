package org.bytedeco.pytorch.dataframe.mongo;

/**
 * Unchecked failure from {@link Mongo} Data API / SPI operations.
 */
public class MongoException extends RuntimeException {
    private final int status;
    private final String operation;

    public MongoException(String message) {
        this(message, null, -1, null);
    }

    public MongoException(String message, Throwable cause) {
        this(message, cause, -1, null);
    }

    public MongoException(String message, int status, String operation) {
        this(message, null, status, operation);
    }

    public MongoException(String message, Throwable cause, int status, String operation) {
        super(message, cause);
        this.status = status;
        this.operation = operation;
    }

    public int status() { return status; }
    public String operation() { return operation; }
}
