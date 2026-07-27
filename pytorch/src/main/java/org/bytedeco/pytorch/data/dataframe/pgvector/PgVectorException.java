package org.bytedeco.pytorch.data.dataframe.pgvector;

/**
 * Unchecked failure from {@link PgVector} JDBC / SPI operations.
 */
public class PgVectorException extends RuntimeException {
    private final String sqlState;
    private final String operation;

    public PgVectorException(String message) {
        this(message, null, null, null);
    }

    public PgVectorException(String message, Throwable cause) {
        this(message, cause, null, null);
    }

    public PgVectorException(String message, Throwable cause, String sqlState, String operation) {
        super(message, cause);
        this.sqlState = sqlState;
        this.operation = operation;
    }

    public String sqlState() { return sqlState; }
    public String operation() { return operation; }
}
