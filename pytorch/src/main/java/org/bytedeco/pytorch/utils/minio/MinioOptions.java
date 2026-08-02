package org.bytedeco.pytorch.utils.minio;

import java.net.URI;
import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable options for MinIO / S3-compatible object storage and DataFrame I/O.
 *
 * <pre>{@code
 * MinioOptions opts = MinioOptions.builder()
 *     .endpoint("http://127.0.0.1:9000")
 *     .accessKey("minioadmin")
 *     .secretKey("minioadmin")
 *     .bucket("datasets")
 *     .objectKey("train/features.jsonl")
 *     .contentType("application/x-ndjson")
 *     .compression(Compression.ZSTD)
 *     .partSize(8L * 1024 * 1024)
 *     .build();
 * df.toMinio(m, opts);
 * }</pre>
 *
 * <p>Also accepts URI forms:
 * {@code minio://access:secret@host:9000/bucket/key?region=us-east-1&secure=false}
 * and {@code s3://bucket/key}.
 */
public final class MinioOptions {

    /** Payload layout when serializing a DataFrame to/from an object. */
    public enum Format {
        /** One JSON object per line (default, row-oriented). */
        JSONL,
        /** Single JSON array of records. */
        JSON,
        /** Arrow-backed parquet file written via {@code DataFrame.writeParquet}. */
        PARQUET,
        /** CSV with header. */
        CSV,
        /** Opaque bytes (BinaryData / MediaData / raw upload). */
        BYTES
    }

    public enum Compression {
        NONE("none"),
        GZIP("gzip"),
        SNAPPY("snappy"),
        ZSTD("zstd");

        private final String value;

        Compression(String value) {
            this.value = value;
        }

        public String value() {
            return value;
        }

        public static Compression from(String s) {
            if (s == null || s.isBlank()) return NONE;
            return switch (s.trim().toLowerCase(Locale.ROOT)) {
                case "gzip", "gz" -> GZIP;
                case "snappy" -> SNAPPY;
                case "zstd", "zst" -> ZSTD;
                default -> NONE;
            };
        }
    }

    public enum IfExists {
        /** Overwrite object (default). */
        REPLACE,
        /** Skip write when object already exists. */
        SKIP,
        /** Fail when object already exists. */
        FAIL
    }

    /** Default multipart part size: 8 MiB (MinIO minimum is 5 MiB). */
    public static final long DEFAULT_PART_SIZE = 8L * 1024 * 1024;
    public static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(60);
    public static final int DEFAULT_MAX_RETRIES = 3;

    private final String endpoint;
    private final String accessKey;
    private final String secretKey;
    private final String sessionToken;
    private final String region;
    private final boolean secure;
    private final boolean disableBucketLookup;
    private final String proxy;
    private final Duration timeout;
    private final int maxRetries;
    private final String bucket;
    private final String objectKey;
    private final String prefix;
    private final String versionId;
    private final String contentType;
    private final Compression compression;
    private final Format format;
    private final String storageClass;
    private final Map<String, String> userMetadata;
    private final Map<String, String> tags;
    private final Map<String, String> headers;
    private final long partSize;
    private final Long objectSize;
    private final long offset;
    private final Long length;
    private final String matchETag;
    private final String notMatchETag;
    private final boolean ensureBucket;
    private final IfExists ifExists;
    private final boolean recursive;
    private final boolean includeVersions;
    private final int maxKeys;
    private final boolean legalHold;
    private final Integer retentionDays;
    private final String retentionMode;
    private final boolean autoDetectFormat;
    private final int selectLimit;
    private final String selectExpression;
    private final String selectInputSerialization;
    private final String selectOutputSerialization;

    private MinioOptions(Builder b) {
        this.endpoint = b.endpoint == null || b.endpoint.isBlank() ? "http://127.0.0.1:9000" : b.endpoint.trim();
        this.accessKey = b.accessKey == null ? "minioadmin" : b.accessKey;
        this.secretKey = b.secretKey == null ? "minioadmin" : b.secretKey;
        this.sessionToken = b.sessionToken;
        this.region = b.region == null || b.region.isBlank() ? "us-east-1" : b.region;
        this.secure = b.secure;
        this.disableBucketLookup = b.disableBucketLookup;
        this.proxy = b.proxy;
        this.timeout = b.timeout == null ? DEFAULT_TIMEOUT : b.timeout;
        this.maxRetries = Math.max(0, b.maxRetries);
        this.bucket = b.bucket;
        this.objectKey = b.objectKey;
        this.prefix = b.prefix == null ? "" : b.prefix;
        this.versionId = b.versionId;
        this.contentType = b.contentType;
        this.compression = b.compression == null ? Compression.NONE : b.compression;
        this.format = b.format == null ? Format.JSONL : b.format;
        this.storageClass = b.storageClass;
        this.userMetadata = b.userMetadata == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.userMetadata));
        this.tags = b.tags == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
        this.headers = b.headers == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.headers));
        this.partSize = b.partSize <= 0 ? DEFAULT_PART_SIZE : b.partSize;
        this.objectSize = b.objectSize;
        this.offset = Math.max(0L, b.offset);
        this.length = b.length;
        this.matchETag = b.matchETag;
        this.notMatchETag = b.notMatchETag;
        this.ensureBucket = b.ensureBucket;
        this.ifExists = b.ifExists == null ? IfExists.REPLACE : b.ifExists;
        this.recursive = b.recursive;
        this.includeVersions = b.includeVersions;
        this.maxKeys = b.maxKeys <= 0 ? 1000 : b.maxKeys;
        this.legalHold = b.legalHold;
        this.retentionDays = b.retentionDays;
        this.retentionMode = b.retentionMode;
        this.autoDetectFormat = b.autoDetectFormat;
        this.selectLimit = b.selectLimit;
        this.selectExpression = b.selectExpression;
        this.selectInputSerialization = b.selectInputSerialization;
        this.selectOutputSerialization = b.selectOutputSerialization;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static MinioOptions defaults() {
        return builder().build();
    }

    public static MinioOptions bucket(String bucket, String objectKey) {
        return builder().bucket(bucket).objectKey(objectKey).build();
    }

    public static MinioOptions of(String endpoint, String accessKey, String secretKey) {
        return builder().endpoint(endpoint).accessKey(accessKey).secretKey(secretKey).build();
    }

    public Builder toBuilder() {
        return new Builder()
                .endpoint(endpoint)
                .accessKey(accessKey)
                .secretKey(secretKey)
                .sessionToken(sessionToken)
                .region(region)
                .secure(secure)
                .disableBucketLookup(disableBucketLookup)
                .proxy(proxy)
                .timeout(timeout)
                .maxRetries(maxRetries)
                .bucket(bucket)
                .objectKey(objectKey)
                .prefix(prefix)
                .versionId(versionId)
                .contentType(contentType)
                .compression(compression)
                .format(format)
                .storageClass(storageClass)
                .userMetadata(userMetadata.isEmpty() ? null : new LinkedHashMap<>(userMetadata))
                .tags(tags.isEmpty() ? null : new LinkedHashMap<>(tags))
                .headers(headers.isEmpty() ? null : new LinkedHashMap<>(headers))
                .partSize(partSize)
                .objectSize(objectSize == null ? -1L : objectSize)
                .offset(offset)
                .length(length == null ? -1L : length)
                .matchETag(matchETag)
                .notMatchETag(notMatchETag)
                .ensureBucket(ensureBucket)
                .ifExists(ifExists)
                .recursive(recursive)
                .includeVersions(includeVersions)
                .maxKeys(maxKeys)
                .legalHold(legalHold)
                .retentionDays(retentionDays == null ? -1 : retentionDays)
                .retentionMode(retentionMode)
                .autoDetectFormat(autoDetectFormat)
                .selectLimit(selectLimit)
                .selectExpression(selectExpression)
                .selectInputSerialization(selectInputSerialization)
                .selectOutputSerialization(selectOutputSerialization);
    }

    public MinioOptions withBucket(String bucket) {
        return toBuilder().bucket(bucket).build();
    }

    public MinioOptions withObjectKey(String objectKey) {
        return toBuilder().objectKey(objectKey).build();
    }

    public MinioOptions withPrefix(String prefix) {
        return toBuilder().prefix(prefix).build();
    }

    public String endpoint() {
        return endpoint;
    }

    public String accessKey() {
        return accessKey;
    }

    public String secretKey() {
        return secretKey;
    }

    public String sessionToken() {
        return sessionToken;
    }

    public String region() {
        return region;
    }

    public boolean secure() {
        return secure;
    }

    public boolean disableBucketLookup() {
        return disableBucketLookup;
    }

    public String proxy() {
        return proxy;
    }

    public Duration timeout() {
        return timeout;
    }

    public int maxRetries() {
        return maxRetries;
    }

    public String bucket() {
        return bucket;
    }

    public String objectKey() {
        return objectKey;
    }

    public String prefix() {
        return prefix;
    }

    public String versionId() {
        return versionId;
    }

    public String contentType() {
        return contentType;
    }

    public Compression compression() {
        return compression;
    }

    public Format format() {
        return format;
    }

    public String storageClass() {
        return storageClass;
    }

    public Map<String, String> userMetadata() {
        return userMetadata;
    }

    public Map<String, String> tags() {
        return tags;
    }

    public Map<String, String> headers() {
        return headers;
    }

    public long partSize() {
        return partSize;
    }

    public Long objectSize() {
        return objectSize;
    }

    public long offset() {
        return offset;
    }

    public Long length() {
        return length;
    }

    public String matchETag() {
        return matchETag;
    }

    public String notMatchETag() {
        return notMatchETag;
    }

    public boolean ensureBucket() {
        return ensureBucket;
    }

    public IfExists ifExists() {
        return ifExists;
    }

    public boolean recursive() {
        return recursive;
    }

    public boolean includeVersions() {
        return includeVersions;
    }

    public int maxKeys() {
        return maxKeys;
    }

    public boolean legalHold() {
        return legalHold;
    }

    public Integer retentionDays() {
        return retentionDays;
    }

    public String retentionMode() {
        return retentionMode;
    }

    public boolean autoDetectFormat() {
        return autoDetectFormat;
    }

    public int selectLimit() {
        return selectLimit;
    }

    public String selectExpression() {
        return selectExpression;
    }

    public String selectInputSerialization() {
        return selectInputSerialization;
    }

    public String selectOutputSerialization() {
        return selectOutputSerialization;
    }

    /** Resolve content-type, falling back by format / compression. */
    public String resolvedContentType() {
        if (contentType != null && !contentType.isBlank()) return contentType;
        String base = switch (format) {
            case JSONL -> "application/x-ndjson";
            case JSON -> "application/json";
            case PARQUET -> "application/vnd.apache.parquet";
            case CSV -> "text/csv";
            case BYTES -> "application/octet-stream";
        };
        return switch (compression) {
            case GZIP -> base + "+gzip";
            case ZSTD -> base + "+zstd";
            case SNAPPY -> base + "+snappy";
            case NONE -> base;
        };
    }

    /**
     * Parse {@code minio://}/{@code s3://}/{@code http(s)://} URIs.
     *
     * <p>Examples:
     * <ul>
     *   <li>{@code minio://minioadmin:minioadmin@127.0.0.1:9000/bucket/key}</li>
     *   <li>{@code s3://bucket/prefix/object.parquet}</li>
     *   <li>{@code http://127.0.0.1:9000/bucket/key?accessKey=a&secretKey=s}</li>
     * </ul>
     */
    public static MinioOptions fromUri(String uri) {
        Objects.requireNonNull(uri, "uri");
        String raw = uri.trim();
        Builder b = builder();

        try {
            String normalized = raw;
            if (normalized.startsWith("minio://")) {
                normalized = "http://" + normalized.substring("minio://".length());
                b.secure(false);
            } else if (normalized.startsWith("s3://")) {
                // s3://bucket/key — endpoint left default / env
                String rest = normalized.substring("s3://".length());
                int slash = rest.indexOf('/');
                if (slash < 0) {
                    b.bucket(rest);
                } else {
                    b.bucket(rest.substring(0, slash));
                    b.objectKey(rest.substring(slash + 1));
                }
                applyQuery(b, rest.contains("?") ? rest.substring(rest.indexOf('?') + 1) : null);
                return b.build();
            } else if (normalized.startsWith("https://")) {
                b.secure(true);
            } else if (normalized.startsWith("http://")) {
                b.secure(false);
            }

            URI u = URI.create(normalized);
            String userInfo = u.getUserInfo();
            if (userInfo != null && !userInfo.isBlank()) {
                int colon = userInfo.indexOf(':');
                if (colon >= 0) {
                    b.accessKey(userInfo.substring(0, colon));
                    b.secretKey(userInfo.substring(colon + 1));
                } else {
                    b.accessKey(userInfo);
                }
            }

            String host = u.getHost();
            int port = u.getPort();
            String scheme = b.secure ? "https" : (u.getScheme() == null ? "http" : u.getScheme());
            if (host != null) {
                if (port > 0) b.endpoint(scheme + "://" + host + ":" + port);
                else b.endpoint(scheme + "://" + host);
            }

            String path = u.getPath();
            if (path != null && path.length() > 1) {
                String p = path.startsWith("/") ? path.substring(1) : path;
                int slash = p.indexOf('/');
                if (slash < 0) {
                    b.bucket(p);
                } else {
                    b.bucket(p.substring(0, slash));
                    String key = p.substring(slash + 1);
                    if (!key.isEmpty()) b.objectKey(key);
                }
            }
            applyQuery(b, u.getQuery());
            return b.build();
        } catch (Exception e) {
            throw new MinioException("invalid minio uri: " + uri, e, "fromUri", null, null);
        }
    }

    public static MinioOptions fromMap(Map<String, Object> cfg) {
        if (cfg == null || cfg.isEmpty()) return defaults();
        Builder b = builder();
        if (cfg.get("endpoint") != null) b.endpoint(String.valueOf(cfg.get("endpoint")));
        else if (cfg.get("url") != null) b.endpoint(String.valueOf(cfg.get("url")));
        if (cfg.get("accessKey") != null) b.accessKey(String.valueOf(cfg.get("accessKey")));
        else if (cfg.get("access_key") != null) b.accessKey(String.valueOf(cfg.get("access_key")));
        if (cfg.get("secretKey") != null) b.secretKey(String.valueOf(cfg.get("secretKey")));
        else if (cfg.get("secret_key") != null) b.secretKey(String.valueOf(cfg.get("secret_key")));
        if (cfg.get("sessionToken") != null) b.sessionToken(String.valueOf(cfg.get("sessionToken")));
        if (cfg.get("region") != null) b.region(String.valueOf(cfg.get("region")));
        if (cfg.get("bucket") != null) b.bucket(String.valueOf(cfg.get("bucket")));
        if (cfg.get("objectKey") != null) b.objectKey(String.valueOf(cfg.get("objectKey")));
        else if (cfg.get("key") != null) b.objectKey(String.valueOf(cfg.get("key")));
        if (cfg.get("prefix") != null) b.prefix(String.valueOf(cfg.get("prefix")));
        if (cfg.get("contentType") != null) b.contentType(String.valueOf(cfg.get("contentType")));
        if (cfg.get("compression") != null) b.compression(Compression.from(String.valueOf(cfg.get("compression"))));
        if (cfg.get("format") != null) {
            try {
                b.format(Format.valueOf(String.valueOf(cfg.get("format")).trim().toUpperCase(Locale.ROOT)));
            } catch (Exception ignored) {}
        }
        Object part = cfg.get("partSize");
        if (part instanceof Number n) b.partSize(n.longValue());
        Object secure = cfg.get("secure");
        if (secure instanceof Boolean bo) b.secure(bo);
        else if (secure != null) b.secure(Boolean.parseBoolean(String.valueOf(secure)));
        Object useSsl = cfg.get("useSSL");
        if (useSsl instanceof Boolean bo) b.secure(bo);
        Object ensure = cfg.get("ensureBucket");
        if (ensure instanceof Boolean bo) b.ensureBucket(bo);
        return b.build();
    }

    private static void applyQuery(Builder b, String query) {
        if (query == null || query.isBlank()) return;
        for (String pair : query.split("&")) {
            if (pair.isBlank()) continue;
            int eq = pair.indexOf('=');
            String k = eq < 0 ? pair : pair.substring(0, eq);
            String v = eq < 0 ? "" : pair.substring(eq + 1);
            k = k.trim().toLowerCase(Locale.ROOT);
            switch (k) {
                case "accesskey", "access_key", "ak" -> b.accessKey(v);
                case "secretkey", "secret_key", "sk" -> b.secretKey(v);
                case "sessiontoken", "token" -> b.sessionToken(v);
                case "region" -> b.region(v);
                case "secure", "usessl", "ssl" -> b.secure(Boolean.parseBoolean(v));
                case "bucket" -> b.bucket(v);
                case "key", "object", "objectkey" -> b.objectKey(v);
                case "prefix" -> b.prefix(v);
                case "compression", "compress" -> b.compression(Compression.from(v));
                case "format" -> {
                    try {
                        b.format(Format.valueOf(v.trim().toUpperCase(Locale.ROOT)));
                    } catch (Exception ignored) {}
                }
                case "contenttype", "content_type" -> b.contentType(v);
                case "partsize", "part_size" -> {
                    try {
                        b.partSize(Long.parseLong(v));
                    } catch (Exception ignored) {}
                }
                case "ensurebucket" -> b.ensureBucket(Boolean.parseBoolean(v));
                case "disablebucketlookup" -> b.disableBucketLookup(Boolean.parseBoolean(v));
                case "proxy" -> b.proxy(v);
                default -> { /* ignore unknown */ }
            }
        }
    }

    public static final class Builder {
        private String endpoint = "http://127.0.0.1:9000";
        private String accessKey = "minioadmin";
        private String secretKey = "minioadmin";
        private String sessionToken;
        private String region = "us-east-1";
        private boolean secure = false;
        private boolean disableBucketLookup = false;
        private String proxy;
        private Duration timeout = DEFAULT_TIMEOUT;
        private int maxRetries = DEFAULT_MAX_RETRIES;
        private String bucket;
        private String objectKey;
        private String prefix = "";
        private String versionId;
        private String contentType;
        private Compression compression = Compression.NONE;
        private Format format = Format.JSONL;
        private String storageClass;
        private Map<String, String> userMetadata;
        private Map<String, String> tags;
        private Map<String, String> headers;
        private long partSize = DEFAULT_PART_SIZE;
        private Long objectSize;
        private long offset;
        private Long length;
        private String matchETag;
        private String notMatchETag;
        private boolean ensureBucket = true;
        private IfExists ifExists = IfExists.REPLACE;
        private boolean recursive = true;
        private boolean includeVersions = false;
        private int maxKeys = 1000;
        private boolean legalHold = false;
        private Integer retentionDays;
        private String retentionMode;
        private boolean autoDetectFormat = true;
        private int selectLimit = 0;
        private String selectExpression;
        private String selectInputSerialization;
        private String selectOutputSerialization;

        public Builder endpoint(String v) {
            this.endpoint = v;
            if (v != null) {
                String s = v.trim().toLowerCase(Locale.ROOT);
                if (s.startsWith("https://")) this.secure = true;
                else if (s.startsWith("http://")) this.secure = false;
            }
            return this;
        }

        public Builder accessKey(String v) { this.accessKey = v; return this; }
        public Builder secretKey(String v) { this.secretKey = v; return this; }
        public Builder sessionToken(String v) { this.sessionToken = v; return this; }
        public Builder region(String v) { this.region = v; return this; }
        public Builder secure(boolean v) { this.secure = v; return this; }
        public Builder useSSL(boolean v) { this.secure = v; return this; }
        public Builder disableBucketLookup(boolean v) { this.disableBucketLookup = v; return this; }
        public Builder proxy(String v) { this.proxy = v; return this; }
        public Builder timeout(Duration v) { this.timeout = v; return this; }
        public Builder maxRetries(int v) { this.maxRetries = v; return this; }
        public Builder bucket(String v) { this.bucket = v; return this; }
        public Builder objectKey(String v) { this.objectKey = v; return this; }
        public Builder key(String v) { this.objectKey = v; return this; }
        public Builder prefix(String v) { this.prefix = v; return this; }
        public Builder versionId(String v) { this.versionId = v; return this; }
        public Builder contentType(String v) { this.contentType = v; return this; }
        public Builder compression(Compression v) { this.compression = v; return this; }
        public Builder format(Format v) { this.format = v; return this; }
        public Builder storageClass(String v) { this.storageClass = v; return this; }
        public Builder userMetadata(Map<String, String> v) { this.userMetadata = v; return this; }
        public Builder tags(Map<String, String> v) { this.tags = v; return this; }
        public Builder headers(Map<String, String> v) { this.headers = v; return this; }
        public Builder partSize(long v) { this.partSize = v; return this; }
        public Builder objectSize(long v) { this.objectSize = v < 0 ? null : v; return this; }
        public Builder offset(long v) { this.offset = v; return this; }
        public Builder length(long v) { this.length = v < 0 ? null : v; return this; }
        public Builder range(long offset, long length) {
            this.offset = offset;
            this.length = length < 0 ? null : length;
            return this;
        }
        public Builder matchETag(String v) { this.matchETag = v; return this; }
        public Builder notMatchETag(String v) { this.notMatchETag = v; return this; }
        public Builder ensureBucket(boolean v) { this.ensureBucket = v; return this; }
        public Builder ifExists(IfExists v) { this.ifExists = v; return this; }
        public Builder recursive(boolean v) { this.recursive = v; return this; }
        public Builder includeVersions(boolean v) { this.includeVersions = v; return this; }
        public Builder maxKeys(int v) { this.maxKeys = v; return this; }
        public Builder legalHold(boolean v) { this.legalHold = v; return this; }
        public Builder retentionDays(int v) { this.retentionDays = v < 0 ? null : v; return this; }
        public Builder retentionMode(String v) { this.retentionMode = v; return this; }
        public Builder autoDetectFormat(boolean v) { this.autoDetectFormat = v; return this; }
        public Builder selectLimit(int v) { this.selectLimit = v; return this; }
        public Builder selectExpression(String v) { this.selectExpression = v; return this; }
        public Builder selectInputSerialization(String v) { this.selectInputSerialization = v; return this; }
        public Builder selectOutputSerialization(String v) { this.selectOutputSerialization = v; return this; }

        public Builder putUserMeta(String k, String v) {
            if (this.userMetadata == null) this.userMetadata = new LinkedHashMap<>();
            this.userMetadata.put(k, v);
            return this;
        }

        public Builder putTag(String k, String v) {
            if (this.tags == null) this.tags = new LinkedHashMap<>();
            this.tags.put(k, v);
            return this;
        }

        public MinioOptions build() {
            return new MinioOptions(this);
        }
    }
}
