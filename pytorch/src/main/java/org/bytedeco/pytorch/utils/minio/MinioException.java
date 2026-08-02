package org.bytedeco.pytorch.utils.minio;

/**
 * Unchecked failure from MinIO adapter operations
 * (connection, put/get, bucket, versioning, select, or stream I/O).
 */
public class MinioException extends RuntimeException {

    private final String operation;
    private final String bucket;
    private final String objectKey;
    private final String errorCode;

    public MinioException(String message) {
        this(message, null, null, null, null, null);
    }

    public MinioException(String message, Throwable cause) {
        this(message, cause, null, null, null, null);
    }

    public MinioException(String message, Throwable cause, String operation) {
        this(message, cause, operation, null, null, null);
    }

    public MinioException(String message, Throwable cause, String operation,
                          String bucket, String objectKey) {
        this(message, cause, operation, bucket, objectKey, null);
    }

    public MinioException(String message, Throwable cause, String operation,
                          String bucket, String objectKey, String errorCode) {
        super(message, cause);
        this.operation = operation;
        this.bucket = bucket;
        this.objectKey = objectKey;
        this.errorCode = errorCode;
    }

    public String operation() {
        return operation;
    }

    public String bucket() {
        return bucket;
    }

    public String objectKey() {
        return objectKey;
    }

    public String errorCode() {
        return errorCode;
    }

    /** Map common MinIO / S3 error codes to a short category. */
    public static String mapErrorCode(Throwable t) {
        if (t == null) return null;
        String msg = t.getMessage();
        if (msg == null) msg = t.getClass().getSimpleName();
        String lower = msg.toLowerCase();
        if (lower.contains("nosuchbucket") || lower.contains("no such bucket")) return "NoSuchBucket";
        if (lower.contains("nosuchkey") || lower.contains("no such key") || lower.contains("not found")) return "NoSuchKey";
        if (lower.contains("accessdenied") || lower.contains("access denied") || lower.contains("403")) return "AccessDenied";
        if (lower.contains("invalidaccesskey") || lower.contains("invalid access")) return "InvalidAccessKeyId";
        if (lower.contains("signature") || lower.contains("403 forbidden")) return "SignatureDoesNotMatch";
        if (lower.contains("bucketalreadyowned") || lower.contains("already owned")) return "BucketAlreadyOwnedByYou";
        if (lower.contains("bucketalreadyexists")) return "BucketAlreadyExists";
        if (lower.contains("timeout") || lower.contains("timed out")) return "Timeout";
        if (lower.contains("connection") || lower.contains("connect")) return "ConnectionFailed";
        return t.getClass().getSimpleName();
    }

    public static MinioException wrap(String operation, String bucket, String objectKey, Throwable t) {
        String code = mapErrorCode(t);
        String msg = operation + " failed"
                + (bucket != null ? " bucket=" + bucket : "")
                + (objectKey != null ? " key=" + objectKey : "")
                + (code != null ? " code=" + code : "")
                + ": " + (t.getMessage() == null ? t.getClass().getSimpleName() : t.getMessage());
        return new MinioException(msg, t, operation, bucket, objectKey, code);
    }
}
