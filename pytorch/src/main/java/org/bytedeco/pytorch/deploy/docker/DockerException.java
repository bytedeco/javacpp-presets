package org.bytedeco.pytorch.deploy.docker;

/**
 * Unchecked exception for Docker CLI / Engine / Compose failures.
 */
public class DockerException extends RuntimeException {

    private final int exitCode;
    private final int httpStatus;
    private final String operation;

    public DockerException(String message) {
        this(message, null, -1, -1, null);
    }

    public DockerException(String message, Throwable cause) {
        this(message, cause, -1, -1, null);
    }

    public DockerException(String message, int exitCode, String operation) {
        this(message, null, exitCode, -1, operation);
    }

    public DockerException(String message, Throwable cause, int exitCode, int httpStatus, String operation) {
        super(message, cause);
        this.exitCode = exitCode;
        this.httpStatus = httpStatus;
        this.operation = operation;
    }

    public static DockerException ofExit(String operation, int exitCode, String output) {
        String body = output == null ? "" : output.trim();
        if (body.length() > 800) body = body.substring(0, 800) + "…";
        return new DockerException(
                "docker " + operation + " failed exit=" + exitCode
                        + (body.isEmpty() ? "" : ": " + body),
                null, exitCode, -1, operation);
    }

    public static DockerException ofHttp(String operation, int status, String body) {
        String b = body == null ? "" : body.trim();
        if (b.length() > 500) b = b.substring(0, 500) + "…";
        return new DockerException(
                "docker engine " + operation + " HTTP " + status
                        + (b.isEmpty() ? "" : ": " + b),
                null, -1, status, operation);
    }

    public int exitCode() { return exitCode; }
    public int httpStatus() { return httpStatus; }
    public String operation() { return operation; }
}
