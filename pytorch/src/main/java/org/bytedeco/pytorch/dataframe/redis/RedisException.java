package org.bytedeco.pytorch.dataframe.redis;

/**
 * Unchecked failure from {@link Redis} / RESP operations
 * (connection, protocol, or server error reply).
 */
public class RedisException extends RuntimeException {
    private final String command;

    public RedisException(String message) {
        this(message, null, null);
    }

    public RedisException(String message, Throwable cause) {
        this(message, cause, null);
    }

    public RedisException(String message, Throwable cause, String command) {
        super(message, cause);
        this.command = command;
    }

    public String command() {
        return command;
    }
}
