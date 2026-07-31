package org.bytedeco.pytorch.deploy.k8s;

/**
 * Unchecked exception for kubectl / apiserver failures.
 */
public class K8sException extends RuntimeException {

    private final int exitCode;
    private final int httpStatus;
    private final String operation;
    private final String resource;

    public K8sException(String message) {
        this(message, null, -1, -1, null, null);
    }

    public K8sException(String message, Throwable cause) {
        this(message, cause, -1, -1, null, null);
    }

    public K8sException(String message, int exitCode, String operation) {
        this(message, null, exitCode, -1, operation, null);
    }

    public K8sException(
            String message, Throwable cause, int exitCode, int httpStatus,
            String operation, String resource) {
        super(message, cause);
        this.exitCode = exitCode;
        this.httpStatus = httpStatus;
        this.operation = operation;
        this.resource = resource;
    }

    public static K8sException ofExit(String operation, int exitCode, String output) {
        String body = output == null ? "" : output.trim();
        if (body.length() > 800) body = body.substring(0, 800) + "…";
        return new K8sException(
                "kubectl " + operation + " failed exit=" + exitCode
                        + (body.isEmpty() ? "" : ": " + body),
                null, exitCode, -1, operation, null);
    }

    public static K8sException ofHttp(String operation, int status, String body) {
        String b = body == null ? "" : body.trim();
        if (b.length() > 500) b = b.substring(0, 500) + "…";
        return new K8sException(
                "k8s api " + operation + " HTTP " + status
                        + (b.isEmpty() ? "" : ": " + b),
                null, -1, status, operation, null);
    }

    public int exitCode() { return exitCode; }
    public int httpStatus() { return httpStatus; }
    public String operation() { return operation; }
    public String resource() { return resource; }
}
