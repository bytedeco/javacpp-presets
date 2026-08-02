package org.bytedeco.pytorch.utils.minio;

import com.github.luben.zstd.Zstd;
import io.minio.BucketExistsArgs;
import okhttp3.Headers;
import io.minio.ComposeObjectArgs;
import io.minio.CopyObjectArgs;
import io.minio.DownloadObjectArgs;
import io.minio.GetObjectArgs;
import io.minio.GetObjectAttributesArgs;
import io.minio.GetObjectAttributesResponse;
import io.minio.GetObjectResponse;
import io.minio.GetObjectTagsArgs;
import io.minio.GetPresignedObjectUrlArgs;
import io.minio.Http;
import io.minio.ListObjectsArgs;
import io.minio.MakeBucketArgs;
import io.minio.MinioClient;
import io.minio.ObjectWriteResponse;
import io.minio.PutObjectArgs;
import io.minio.RemoveBucketArgs;
import io.minio.RemoveObjectArgs;
import io.minio.RemoveObjectsArgs;
import io.minio.Result;
import io.minio.SelectObjectContentArgs;
import io.minio.SelectResponseStream;
import io.minio.ServerSideEncryption;
import io.minio.SetObjectTagsArgs;
import io.minio.SourceObject;
import io.minio.StatObjectArgs;
import io.minio.StatObjectResponse;
import io.minio.UploadObjectArgs;
import io.minio.messages.Item;
import io.minio.messages.Tags;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.csv.CsvOptions;
import org.bytedeco.pytorch.dataframe.csv.CsvWriter;
import org.bytedeco.pytorch.dataframe.dtype.BinaryData;
import org.bytedeco.pytorch.utils.json.Json;
import org.xerial.snappy.Snappy;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Enterprise MinIO / S3-compatible façade for DataFrame binary I/O, bucket admin,
 * versioning, multipart upload, pre-signed URLs, and SelectObjectContent.
 *
 * <p>Built on official {@code io.minio:minio:9.0.3}. Defaults target dataset / feature-store
 * object pipelines: JSONL row dumps, parquet snapshots, BinaryData / media blobs.
 *
 * <pre>{@code
 * try (Minio m = Minio.connect("http://127.0.0.1:9000", "minioadmin", "minioadmin")) {
 *     m.ensureBucket("datasets");
 *     df.toMinio(m, MinioOptions.builder()
 *         .bucket("datasets")
 *         .objectKey("train/features.jsonl")
 *         .format(MinioOptions.Format.JSONL)
 *         .compression(MinioOptions.Compression.ZSTD)
 *         .build());
 *     DataFrame back = DataFrame.readMinio(m, MinioOptions.bucket("datasets", "train/features.jsonl"));
 * }
 * }</pre>
 *
 * @see MinioOptions
 * @see MinioFile
 * @see MinioStream
 * @see MinioBucket
 * @see MinioVersion
 * @see MinioMetrics
 */
public final class Minio implements Closeable {

    public static final String VERSION = "9.0.3";
    public static final String APP_NAME = "jnitorch-minio";

    private final MinioClient client;
    private final MinioOptions options;
    private final MinioMetrics metrics;
    private final boolean ownClient;
    private volatile boolean closed;

    private Minio(MinioClient client, MinioOptions options, boolean ownClient) {
        this.client = Objects.requireNonNull(client, "client");
        this.options = options == null ? MinioOptions.defaults() : options;
        this.metrics = new MinioMetrics();
        this.ownClient = ownClient;
        try {
            this.client.setAppInfo(APP_NAME, VERSION);
            Duration t = this.options.timeout();
            if (t != null && !t.isZero() && !t.isNegative()) {
                long ms = Math.max(1_000L, t.toMillis());
                this.client.setTimeout(ms, ms, ms);
            }
            if (this.options.maxRetries() > 0) {
                this.client.setRetry(java.util.Set.of(408, 429, 500, 502, 503, 504), 200L, this.options.maxRetries());
            }
        } catch (Exception ignored) {
        }
    }

    // ── factories ────────────────────────────────────────────────────────────

    public static Minio connect() {
        return connect(MinioOptions.defaults());
    }

    public static Minio connect(String endpoint, String accessKey, String secretKey) {
        return connect(MinioOptions.builder()
                .endpoint(endpoint)
                .accessKey(accessKey)
                .secretKey(secretKey)
                .build());
    }

    public static Minio connect(String endpoint, String accessKey, String secretKey, String region) {
        return connect(MinioOptions.builder()
                .endpoint(endpoint)
                .accessKey(accessKey)
                .secretKey(secretKey)
                .region(region)
                .build());
    }

    public static Minio connect(MinioOptions options) {
        Objects.requireNonNull(options, "options");
        try {
            MinioClient.Builder b = MinioClient.builder()
                    .endpoint(normalizeEndpoint(options.endpoint()))
                    .credentials(options.accessKey(), options.secretKey());
            if (options.region() != null && !options.region().isBlank()) {
                b.region(options.region());
            }
            MinioClient c = b.build();
            if (options.disableBucketLookup()) {
                try {
                    c.disableVirtualStyleEndpoint();
                } catch (Exception ignored) {}
            }
            return new Minio(c, options, true);
        } catch (Exception e) {
            throw MinioException.wrap("connect", null, null, e);
        }
    }

    public static Minio connectUri(String uri) {
        return connect(MinioOptions.fromUri(uri));
    }

    public static Minio wrap(MinioClient client, MinioOptions options) {
        return new Minio(client, options, false);
    }

    public static Minio fromMap(Map<String, Object> cfg) {
        return connect(MinioOptions.fromMap(cfg));
    }

    public static Minio fromEnv() {
        String endpoint = firstEnv("MINIO_ENDPOINT", "MINIO_URL", "S3_ENDPOINT");
        String access = firstEnv("MINIO_ACCESS_KEY", "MINIO_ROOT_USER", "AWS_ACCESS_KEY_ID");
        String secret = firstEnv("MINIO_SECRET_KEY", "MINIO_ROOT_PASSWORD", "AWS_SECRET_ACCESS_KEY");
        String region = firstEnv("MINIO_REGION", "AWS_REGION", "AWS_DEFAULT_REGION");
        MinioOptions.Builder b = MinioOptions.builder();
        if (endpoint != null) b.endpoint(endpoint);
        if (access != null) b.accessKey(access);
        if (secret != null) b.secretKey(secret);
        if (region != null) b.region(region);
        String bucket = firstEnv("MINIO_BUCKET", "S3_BUCKET");
        if (bucket != null) b.bucket(bucket);
        return connect(b.build());
    }

    // ── accessors ────────────────────────────────────────────────────────────

    public MinioOptions options() { return options; }
    public MinioMetrics metrics() { return metrics; }
    public MinioClient raw() { ensureOpen(); return client; }
    public String endpoint() { return options.endpoint(); }

    public MinioBucket bucketApi() { return new MinioBucket(this); }
    public MinioVersion versionApi() { return new MinioVersion(this); }

    public MinioFile file(String bucket, String objectKey) {
        return MinioFile.of(this, bucket, objectKey);
    }

    public MinioFile file(MinioOptions opts) {
        MinioOptions o = merge(opts);
        requireBucketKey(o, "file");
        return MinioFile.of(this, o.bucket(), o.objectKey(), o.versionId());
    }

    public MinioStream stream(MinioOptions opts) {
        return MinioStream.open(this, merge(opts));
    }

    // ── bucket shortcuts ─────────────────────────────────────────────────────

    public boolean bucketExists(String bucket) {
        ensureOpen();
        try {
            return client.bucketExists(BucketExistsArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("bucketExists", bucket, null, e);
        }
    }

    public void makeBucket(String bucket) {
        makeBucket(bucket, options.region(), false);
    }

    public void makeBucket(String bucket, String region, boolean objectLock) {
        ensureOpen();
        try {
            MakeBucketArgs.Builder b = MakeBucketArgs.builder().bucket(bucket);
            if (region != null && !region.isBlank()) b.region(region);
            if (objectLock) b.objectLock(true);
            client.makeBucket(b.build());
        } catch (Exception e) {
            String code = MinioException.mapErrorCode(e);
            if ("BucketAlreadyOwnedByYou".equals(code) || "BucketAlreadyExists".equals(code)) return;
            throw MinioException.wrap("makeBucket", bucket, null, e);
        }
    }

    public void ensureBucket(String bucket) {
        if (bucket == null || bucket.isBlank()) return;
        if (!bucketExists(bucket)) makeBucket(bucket);
    }

    public void removeBucket(String bucket) {
        ensureOpen();
        try {
            client.removeBucket(RemoveBucketArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("removeBucket", bucket, null, e);
        }
    }

    public List<String> listBuckets() {
        ensureOpen();
        try {
            return client.listBuckets().stream().map(b -> b.name()).collect(Collectors.toList());
        } catch (Exception e) {
            throw MinioException.wrap("listBuckets", null, null, e);
        }
    }

    // ── object put / get / remove ────────────────────────────────────────────

    public ObjectWriteResponse putBytes(String bucket, String objectKey, byte[] data, MinioOptions opts) {
        Objects.requireNonNull(data, "data");
        MinioOptions o = merge(opts).toBuilder().bucket(bucket).objectKey(objectKey).build();
        return putBytes(data, o);
    }

    public ObjectWriteResponse putBytes(byte[] data, MinioOptions opts) {
        Objects.requireNonNull(data, "data");
        MinioOptions o = merge(opts);
        requireBucketKey(o, "putBytes");
        ensureOpen();
        if (o.ensureBucket()) ensureBucket(o.bucket());
        maybeGuardIfExists(o);

        long t0 = System.nanoTime();
        try {
            byte[] payload = maybeCompress(data, o.compression());
            PutObjectArgs.Builder b = PutObjectArgs.builder()
                    .bucket(o.bucket())
                    .object(o.objectKey())
                    .stream(new ByteArrayInputStream(payload), (long) payload.length, o.partSize())
                    .contentType(o.resolvedContentType());
            applyWriteMeta(b, o);
            ObjectWriteResponse resp = client.putObject(b.build());
            double ms = (System.nanoTime() - t0) / 1_000_000.0;
            metrics.recordPut(payload.length, ms, true);
            return resp;
        } catch (SkipWrite sw) {
            throw sw;
        } catch (Exception e) {
            metrics.recordPut(data.length, (System.nanoTime() - t0) / 1_000_000.0, false);
            throw MinioException.wrap("putBytes", o.bucket(), o.objectKey(), e);
        }
    }

    public ObjectWriteResponse putStream(InputStream stream, long objectSize, MinioOptions opts) {
        Objects.requireNonNull(stream, "stream");
        MinioOptions o = merge(opts);
        requireBucketKey(o, "putStream");
        ensureOpen();
        if (o.ensureBucket()) ensureBucket(o.bucket());
        maybeGuardIfExists(o);

        long t0 = System.nanoTime();
        long size = objectSize >= 0 ? objectSize : -1L;
        try {
            PutObjectArgs.Builder b = PutObjectArgs.builder()
                    .bucket(o.bucket())
                    .object(o.objectKey())
                    .stream(stream, size, o.partSize())
                    .contentType(o.resolvedContentType());
            applyWriteMeta(b, o);
            ObjectWriteResponse resp = client.putObject(b.build());
            double ms = (System.nanoTime() - t0) / 1_000_000.0;
            metrics.recordPut(Math.max(0L, size), ms, true);
            return resp;
        } catch (SkipWrite sw) {
            throw sw;
        } catch (Exception e) {
            metrics.recordPut(0, (System.nanoTime() - t0) / 1_000_000.0, false);
            throw MinioException.wrap("putStream", o.bucket(), o.objectKey(), e);
        }
    }

    public ObjectWriteResponse uploadFile(Path path, MinioOptions opts) {
        Objects.requireNonNull(path, "path");
        MinioOptions o = merge(opts);
        requireBucketKey(o, "uploadFile");
        ensureOpen();
        if (o.ensureBucket()) ensureBucket(o.bucket());
        maybeGuardIfExists(o);
        long t0 = System.nanoTime();
        try {
            UploadObjectArgs.Builder b = UploadObjectArgs.builder()
                    .bucket(o.bucket())
                    .object(o.objectKey())
                    .filename(path.toAbsolutePath().toString())
                    .contentType(o.resolvedContentType());
            if (!o.userMetadata().isEmpty()) b.userMetadata(o.userMetadata());
            if (!o.tags().isEmpty()) b.tags(o.tags());
            if (!o.headers().isEmpty()) b.headers(o.headers());
            ObjectWriteResponse resp = client.uploadObject(b.build());
            long bytes = Files.size(path);
            metrics.recordPut(bytes, (System.nanoTime() - t0) / 1_000_000.0, true);
            return resp;
        } catch (SkipWrite sw) {
            throw sw;
        } catch (Exception e) {
            metrics.recordPut(0, (System.nanoTime() - t0) / 1_000_000.0, false);
            throw MinioException.wrap("uploadFile", o.bucket(), o.objectKey(), e);
        }
    }

    public ObjectWriteResponse putBinaryData(BinaryData binary, MinioOptions opts) {
        Objects.requireNonNull(binary, "binary");
        byte[] data = binary.getData();
        MinioOptions o = merge(opts);
        if ((o.objectKey() == null || o.objectKey().isBlank()) && binary.getBinaryName() != null) {
            o = o.toBuilder().objectKey(binary.getBinaryName()).build();
        }
        if (o.contentType() == null) {
            o = o.toBuilder().contentType("application/octet-stream").format(MinioOptions.Format.BYTES).build();
        }
        return putBytes(data, o);
    }

    public byte[] getBytes(MinioOptions opts) {
        MinioOptions o = merge(opts);
        requireBucketKey(o, "getBytes");
        ensureOpen();
        long t0 = System.nanoTime();
        try {
            GetObjectArgs.Builder b = GetObjectArgs.builder()
                    .bucket(o.bucket())
                    .object(o.objectKey());
            if (o.versionId() != null && !o.versionId().isBlank()) b.versionId(o.versionId());
            if (o.length() != null && o.length() > 0) {
                b.offset(o.offset()).length(o.length());
            } else if (o.offset() > 0) {
                b.offset(o.offset());
            }
            if (o.matchETag() != null) b.matchETag(o.matchETag());
            if (o.notMatchETag() != null) b.notMatchETag(o.notMatchETag());

            try (GetObjectResponse resp = client.getObject(b.build())) {
                byte[] raw = resp.readAllBytes();
                byte[] out = maybeDecompress(raw, o.compression(), detectCompression(o, resp.headers()));
                metrics.recordGet(out.length, (System.nanoTime() - t0) / 1_000_000.0, true);
                return out;
            }
        } catch (Exception e) {
            metrics.recordGet(0, (System.nanoTime() - t0) / 1_000_000.0, false);
            throw MinioException.wrap("getBytes", o.bucket(), o.objectKey(), e);
        }
    }

    public byte[] getBytes(String bucket, String objectKey) {
        return getBytes(MinioOptions.builder().bucket(bucket).objectKey(objectKey).build());
    }

    public byte[] getRange(String bucket, String objectKey, long offset, long length) {
        return getBytes(MinioOptions.builder()
                .bucket(bucket).objectKey(objectKey).offset(offset).length(length).build());
    }

    public void downloadObject(Path path, MinioOptions opts) {
        Objects.requireNonNull(path, "path");
        MinioOptions o = merge(opts);
        requireBucketKey(o, "downloadObject");
        ensureOpen();
        long t0 = System.nanoTime();
        try {
            if (path.getParent() != null) Files.createDirectories(path.getParent());
            DownloadObjectArgs.Builder b = DownloadObjectArgs.builder()
                    .bucket(o.bucket())
                    .object(o.objectKey())
                    .filename(path.toAbsolutePath().toString());
            if (o.versionId() != null) b.versionId(o.versionId());
            client.downloadObject(b.build());
            long bytes = Files.exists(path) ? Files.size(path) : 0L;
            metrics.recordGet(bytes, (System.nanoTime() - t0) / 1_000_000.0, true);
        } catch (Exception e) {
            metrics.recordGet(0, (System.nanoTime() - t0) / 1_000_000.0, false);
            throw MinioException.wrap("downloadObject", o.bucket(), o.objectKey(), e);
        }
    }

    public void removeObject(String bucket, String objectKey) {
        removeObject(MinioOptions.builder().bucket(bucket).objectKey(objectKey).build());
    }

    public void removeObject(MinioOptions opts) {
        MinioOptions o = merge(opts);
        requireBucketKey(o, "removeObject");
        ensureOpen();
        try {
            RemoveObjectArgs.Builder b = RemoveObjectArgs.builder()
                    .bucket(o.bucket())
                    .object(o.objectKey());
            if (o.versionId() != null) b.versionId(o.versionId());
            client.removeObject(b.build());
            metrics.recordDelete(true);
        } catch (Exception e) {
            metrics.recordDelete(false);
            throw MinioException.wrap("removeObject", o.bucket(), o.objectKey(), e);
        }
    }

    public void removeObjectVersion(String bucket, String objectKey, String versionId) {
        removeObject(MinioOptions.builder().bucket(bucket).objectKey(objectKey).versionId(versionId).build());
    }

    public int removeObjects(String bucket, List<String> objectKeys) {
        ensureOpen();
        if (objectKeys == null || objectKeys.isEmpty()) return 0;
        try {
            List<io.minio.messages.DeleteRequest.Object> objs = new ArrayList<>(objectKeys.size());
            for (String k : objectKeys) {
                if (k != null && !k.isBlank()) objs.add(new io.minio.messages.DeleteRequest.Object(k));
            }
            Iterable<Result<io.minio.messages.DeleteResult.Error>> results = client.removeObjects(
                    RemoveObjectsArgs.builder().bucket(bucket).objects(objs).build());
            int errors = 0;
            for (Result<io.minio.messages.DeleteResult.Error> r : results) {
                try {
                    if (r.get() != null) {
                        metrics.recordDelete(false);
                        errors++;
                    }
                } catch (Exception e) {
                    metrics.recordDelete(false);
                    errors++;
                }
            }
            int ok = Math.max(0, objectKeys.size() - errors);
            for (int i = 0; i < ok; i++) metrics.recordDelete(true);
            return objectKeys.size();
        } catch (Exception e) {
            throw MinioException.wrap("removeObjects", bucket, null, e);
        }
    }

    public StatObjectResponse statObject(MinioOptions opts) {
        MinioOptions o = merge(opts);
        requireBucketKey(o, "statObject");
        ensureOpen();
        try {
            StatObjectArgs.Builder b = StatObjectArgs.builder().bucket(o.bucket()).object(o.objectKey());
            if (o.versionId() != null) b.versionId(o.versionId());
            return client.statObject(b.build());
        } catch (Exception e) {
            throw MinioException.wrap("statObject", o.bucket(), o.objectKey(), e);
        }
    }

    public boolean objectExists(String bucket, String objectKey) {
        try {
            statObject(MinioOptions.builder().bucket(bucket).objectKey(objectKey).build());
            return true;
        } catch (MinioException e) {
            if ("NoSuchKey".equals(e.errorCode()) || "NoSuchBucket".equals(e.errorCode())) return false;
            String msg = e.getMessage() == null ? "" : e.getMessage().toLowerCase(Locale.ROOT);
            if (msg.contains("not found") || msg.contains("nosuchkey") || msg.contains("no such key")) return false;
            throw e;
        }
    }

    public GetObjectAttributesResponse getObjectAttributes(MinioOptions opts) {
        MinioOptions o = merge(opts);
        requireBucketKey(o, "getObjectAttributes");
        ensureOpen();
        try {
            GetObjectAttributesArgs.Builder b = GetObjectAttributesArgs.builder()
                    .bucket(o.bucket()).object(o.objectKey());
            if (o.versionId() != null) b.versionId(o.versionId());
            return client.getObjectAttributes(b.build());
        } catch (Exception e) {
            throw MinioException.wrap("getObjectAttributes", o.bucket(), o.objectKey(), e);
        }
    }

    // ── copy / compose ───────────────────────────────────────────────────────

    public ObjectWriteResponse copyObject(String srcBucket, String srcKey, String dstBucket, String dstKey) {
        ensureOpen();
        try {
            ObjectWriteResponse resp = client.copyObject(CopyObjectArgs.builder()
                    .bucket(dstBucket)
                    .object(dstKey)
                    .source(SourceObject.builder().bucket(srcBucket).object(srcKey).build())
                    .build());
            metrics.recordCopy(true);
            return resp;
        } catch (Exception e) {
            metrics.recordCopy(false);
            throw MinioException.wrap("copyObject", dstBucket, dstKey, e);
        }
    }

    public ObjectWriteResponse composeObject(String bucket, String objectKey, List<SourceObject> sources) {
        ensureOpen();
        try {
            ObjectWriteResponse resp = client.composeObject(ComposeObjectArgs.builder()
                    .bucket(bucket).object(objectKey).sources(sources).build());
            metrics.recordCopy(true);
            return resp;
        } catch (Exception e) {
            metrics.recordCopy(false);
            throw MinioException.wrap("composeObject", bucket, objectKey, e);
        }
    }

    // ── list ─────────────────────────────────────────────────────────────────

    public List<MinioObjectInfo> listObjects(String bucket, String prefix, boolean recursive) {
        return listObjects(MinioOptions.builder()
                .bucket(bucket).prefix(prefix == null ? "" : prefix).recursive(recursive).build());
    }

    public List<MinioObjectInfo> listObjects(MinioOptions opts) {
        MinioOptions o = merge(opts);
        if (o.bucket() == null || o.bucket().isBlank()) {
            throw new MinioException("bucket required for listObjects", null, "listObjects", null, null);
        }
        ensureOpen();
        List<MinioObjectInfo> out = new ArrayList<>();
        try {
            ListObjectsArgs.Builder b = ListObjectsArgs.builder()
                    .bucket(o.bucket()).recursive(o.recursive()).maxKeys(o.maxKeys());
            if (o.prefix() != null && !o.prefix().isEmpty()) b.prefix(o.prefix());
            if (o.includeVersions()) b.includeVersions(true);
            for (Result<Item> r : client.listObjects(b.build())) {
                Item item = r.get();
                if (item == null) continue;
                out.add(MinioObjectInfo.from(item));
            }
            metrics.recordList(out.size(), true);
            return out;
        } catch (Exception e) {
            metrics.recordList(0, false);
            throw MinioException.wrap("listObjects", o.bucket(), o.prefix(), e);
        }
    }

    // ── DataFrame I/O ────────────────────────────────────────────────────────

    public int writeDataFrame(DataFrame df, MinioOptions opts) {
        Objects.requireNonNull(df, "df");
        MinioOptions o = merge(opts);
        requireBucketKey(o, "writeDataFrame");
        try {
            byte[] payload = encodeDataFrame(df, o);
            MinioOptions putOpts = o.toBuilder()
                    .contentType(o.contentType() != null ? o.contentType() : o.resolvedContentType())
                    .build();
            try {
                putBytes(payload, putOpts);
            } catch (SkipWrite sw) {
                return 0;
            }
            return df.rowCount();
        } catch (MinioException e) {
            throw e;
        } catch (Exception e) {
            throw MinioException.wrap("writeDataFrame", o.bucket(), o.objectKey(), e);
        }
    }

    public int writeDataFrame(DataFrame df) {
        return writeDataFrame(df, options);
    }

    public int writeDataFrame(DataFrame df, String bucket, String objectKey) {
        return writeDataFrame(df, MinioOptions.builder().bucket(bucket).objectKey(objectKey).build());
    }

    public DataFrame readDataFrame(MinioOptions opts) {
        MinioOptions o = merge(opts);
        requireBucketKey(o, "readDataFrame");
        try {
            MinioOptions getOpts = o;
            if (o.compression() == MinioOptions.Compression.NONE && o.autoDetectFormat()) {
                getOpts = o.toBuilder().compression(guessCompressionFromKey(o.objectKey())).build();
            }
            byte[] raw = getBytes(getOpts);
            MinioOptions.Format fmt = o.format();
            if (o.autoDetectFormat()) {
                fmt = guessFormat(o.objectKey(), raw, fmt);
            }
            return decodeDataFrame(raw, fmt, o);
        } catch (MinioException e) {
            throw e;
        } catch (Exception e) {
            throw MinioException.wrap("readDataFrame", o.bucket(), o.objectKey(), e);
        }
    }

    public DataFrame readDataFrame() {
        return readDataFrame(options);
    }

    public DataFrame readDataFrame(String bucket, String objectKey) {
        return readDataFrame(MinioOptions.builder().bucket(bucket).objectKey(objectKey).build());
    }

    public int writeBinaryColumn(DataFrame df, String column, MinioOptions opts) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(column, "column");
        MinioOptions o = merge(opts);
        if (o.bucket() == null) {
            throw new MinioException("bucket required", null, "writeBinaryColumn", null, null);
        }
        String prefix = o.prefix() == null ? "" : o.prefix();
        if (o.objectKey() != null && !o.objectKey().isBlank() && prefix.isEmpty()) {
            prefix = o.objectKey().endsWith("/") ? o.objectKey() : o.objectKey() + "/";
        }
        int n = 0;
        for (int i = 0; i < df.rowCount(); i++) {
            Object cell = df.get(i, column);
            byte[] bytes = cellToBytes(cell);
            if (bytes == null) continue;
            String name = cell instanceof BinaryData bd && bd.getBinaryName() != null
                    ? bd.getBinaryName()
                    : column + "-" + i + ".bin";
            String key = prefix + name;
            putBytes(bytes, o.toBuilder()
                    .objectKey(key)
                    .format(MinioOptions.Format.BYTES)
                    .contentType(o.contentType() != null ? o.contentType() : "application/octet-stream")
                    .build());
            n++;
        }
        return n;
    }

    // ── pre-signed URL ───────────────────────────────────────────────────────

    public String getPresignedObjectUrl(String bucket, String objectKey, Http.Method method, Duration expires) {
        ensureOpen();
        try {
            int expirySec = (int) Math.max(1, (expires == null ? Duration.ofHours(1) : expires).getSeconds());
            return client.getPresignedObjectUrl(GetPresignedObjectUrlArgs.builder()
                    .bucket(bucket)
                    .object(objectKey)
                    .method(method == null ? Http.Method.GET : method)
                    .expiry(expirySec)
                    .build());
        } catch (Exception e) {
            throw MinioException.wrap("getPresignedObjectUrl", bucket, objectKey, e);
        }
    }

    public String getPresignedObjectUrl(String objectKey, String method, Duration expires) {
        String bucket = options.bucket();
        if (bucket == null) {
            throw new MinioException("bucket required", null, "getPresignedObjectUrl", null, objectKey);
        }
        return getPresignedObjectUrl(bucket, objectKey, parseMethod(method), expires);
    }

    // ── tags ─────────────────────────────────────────────────────────────────

    public void setObjectTags(String bucket, String objectKey, Map<String, String> tags) {
        ensureOpen();
        try {
            client.setObjectTags(SetObjectTagsArgs.builder()
                    .bucket(bucket).object(objectKey)
                    .tags(tags == null ? Map.of() : tags).build());
        } catch (Exception e) {
            throw MinioException.wrap("setObjectTags", bucket, objectKey, e);
        }
    }

    public Map<String, String> getObjectTags(String bucket, String objectKey) {
        ensureOpen();
        try {
            Tags t = client.getObjectTags(GetObjectTagsArgs.builder()
                    .bucket(bucket).object(objectKey).build());
            return t == null || t.get() == null ? Map.of() : new LinkedHashMap<>(t.get());
        } catch (Exception e) {
            throw MinioException.wrap("getObjectTags", bucket, objectKey, e);
        }
    }

    // ── select ───────────────────────────────────────────────────────────────

    public byte[] selectObjectContent(MinioOptions opts) {
        MinioOptions o = merge(opts);
        requireBucketKey(o, "selectObjectContent");
        if (o.selectExpression() == null || o.selectExpression().isBlank()) {
            throw new MinioException("selectExpression required", null, "selectObjectContent",
                    o.bucket(), o.objectKey());
        }
        ensureOpen();
        try {
            io.minio.messages.InputSerialization input = buildInputSerialization(o);
            io.minio.messages.OutputSerialization output = buildOutputSerialization(o);
            try (SelectResponseStream stream = client.selectObjectContent(SelectObjectContentArgs.builder()
                    .bucket(o.bucket())
                    .object(o.objectKey())
                    .sqlExpression(o.selectExpression())
                    .inputSerialization(input)
                    .outputSerialization(output)
                    .build())) {
                return stream.readAllBytes();
            }
        } catch (Exception e) {
            throw MinioException.wrap("selectObjectContent", o.bucket(), o.objectKey(), e);
        }
    }

    public DataFrame selectDataFrame(MinioOptions opts) {
        byte[] raw = selectObjectContent(opts);
        MinioOptions o = merge(opts);
        String out = o.selectOutputSerialization() == null
                ? "csv" : o.selectOutputSerialization().toLowerCase(Locale.ROOT);
        try {
            if (out.contains("json")) {
                return decodeDataFrame(raw, MinioOptions.Format.JSONL, o);
            }
            Path tmp = Files.createTempFile("minio-select-", ".csv");
            try {
                Files.write(tmp, raw);
                return DataFrame.readCsv(tmp.toString());
            } finally {
                Files.deleteIfExists(tmp);
            }
        } catch (MinioException e) {
            throw e;
        } catch (Exception e) {
            throw MinioException.wrap("selectDataFrame", o.bucket(), o.objectKey(), e);
        }
    }

    // ── parallel upload ──────────────────────────────────────────────────────

    public int putAll(String bucket, Map<String, byte[]> items, int parallelism) {
        if (items == null || items.isEmpty()) return 0;
        ensureBucket(bucket);
        int threads = Math.max(1, Math.min(parallelism <= 0 ? 4 : parallelism, 32));
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<CompletableFuture<Boolean>> futs = new ArrayList<>(items.size());
            for (Map.Entry<String, byte[]> e : items.entrySet()) {
                futs.add(CompletableFuture.supplyAsync(() -> {
                    try {
                        putBytes(e.getValue(), MinioOptions.builder()
                                .bucket(bucket)
                                .objectKey(e.getKey())
                                .format(MinioOptions.Format.BYTES)
                                .ensureBucket(false)
                                .build());
                        return true;
                    } catch (Exception ex) {
                        return false;
                    }
                }, pool));
            }
            int ok = 0;
            for (CompletableFuture<Boolean> f : futs) {
                if (Boolean.TRUE.equals(f.join())) ok++;
            }
            return ok;
        } finally {
            pool.shutdown();
            try { pool.awaitTermination(60, TimeUnit.SECONDS); }
            catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
        }
    }

    // ── close ────────────────────────────────────────────────────────────────

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (ownClient) {
            try { client.close(); } catch (Exception ignored) {}
        }
    }

    // ── internals ────────────────────────────────────────────────────────────

    void ensureOpen() {
        if (closed) throw new MinioException("Minio client is closed", null, "ensureOpen", null, null);
    }

    MinioOptions merge(MinioOptions override) {
        if (override == null) return options;
        MinioOptions.Builder b = override.toBuilder();
        if (override.bucket() == null && options.bucket() != null) b.bucket(options.bucket());
        return b.build();
    }

    private void requireBucketKey(MinioOptions o, String op) {
        if (o.bucket() == null || o.bucket().isBlank()) {
            throw new MinioException("bucket required for " + op, null, op, null, o.objectKey());
        }
        if (o.objectKey() == null || o.objectKey().isBlank()) {
            throw new MinioException("objectKey required for " + op, null, op, o.bucket(), null);
        }
    }

    private void maybeGuardIfExists(MinioOptions o) {
        if (o.ifExists() == MinioOptions.IfExists.REPLACE) return;
        boolean exists = objectExists(o.bucket(), o.objectKey());
        if (!exists) return;
        if (o.ifExists() == MinioOptions.IfExists.SKIP) {
            throw new SkipWrite("object exists, skip: " + o.bucket() + "/" + o.objectKey());
        }
        throw new MinioException("object already exists: " + o.bucket() + "/" + o.objectKey(),
                null, "ifExists", o.bucket(), o.objectKey(), "ObjectAlreadyExists");
    }

    private void applyWriteMeta(PutObjectArgs.Builder b, MinioOptions o) {
        if (!o.userMetadata().isEmpty()) b.userMetadata(o.userMetadata());
        if (!o.tags().isEmpty()) b.tags(o.tags());
        Map<String, String> h = new LinkedHashMap<>();
        if (!o.headers().isEmpty()) h.putAll(o.headers());
        if (o.storageClass() != null && !o.storageClass().isBlank()) {
            h.putIfAbsent("x-amz-storage-class", o.storageClass());
        }
        if (o.legalHold()) {
            h.putIfAbsent("x-amz-object-lock-legal-hold", "ON");
        }
        if (!h.isEmpty()) b.headers(h);
        try {
            if (h.containsKey("x-amz-server-side-encryption")) {
                b.sse(new ServerSideEncryption.S3());
            }
        } catch (Exception ignored) {}
    }

    static byte[] encodeDataFrame(DataFrame df, MinioOptions o) throws Exception {
        MinioOptions.Format fmt = o.format() == null ? MinioOptions.Format.JSONL : o.format();
        byte[] raw = switch (fmt) {
            case JSONL -> encodeJsonl(df);
            case JSON -> df.toJsonString().getBytes(StandardCharsets.UTF_8);
            case CSV -> encodeCsv(df);
            case PARQUET -> encodeParquet(df);
            case BYTES -> encodeBytesFrame(df);
        };
        return maybeCompress(raw, o.compression());
    }

    static DataFrame decodeDataFrame(byte[] data, MinioOptions.Format fmt, MinioOptions o) throws Exception {
        MinioOptions.Format f = fmt == null ? MinioOptions.Format.JSONL : fmt;
        return switch (f) {
            case JSONL -> decodeJsonl(data);
            case JSON -> DataFrame.readJsonString(new String(data, StandardCharsets.UTF_8));
            case CSV -> decodeCsv(data);
            case PARQUET -> decodeParquet(data);
            case BYTES -> {
                DataFrame df = DataFrame.create();
                df.addColumn("data", org.bytedeco.pytorch.dataframe.Column.DType.BINARY);
                int r = df.addEmptyRow();
                df.set(r, "data", new BinaryData("object", data));
                yield df;
            }
        };
    }

    private static byte[] encodeJsonl(DataFrame df) {
        StringBuilder sb = new StringBuilder(Math.max(256, df.rowCount() * 64));
        for (Map<String, Object> row : df.toRecords()) {
            sb.append(Json.encode(row)).append('\n');
        }
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    private static DataFrame decodeJsonl(byte[] data) throws IOException {
        String text = new String(data, StandardCharsets.UTF_8);
        List<Map<String, Object>> rows = new ArrayList<>();
        for (String line : text.split("\n", -1)) {
            String t = line.trim();
            if (t.isEmpty() || t.startsWith("#")) continue;
            Object v = Json.decode(t);
            if (v instanceof Map<?, ?> m) {
                Map<String, Object> row = new LinkedHashMap<>();
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    if (e.getKey() != null) row.put(String.valueOf(e.getKey()), e.getValue());
                }
                rows.add(row);
            } else {
                Map<String, Object> wrap = new LinkedHashMap<>();
                wrap.put("value", v);
                rows.add(wrap);
            }
        }
        return DataFrame.fromRecords(rows);
    }

    private static byte[] encodeCsv(DataFrame df) throws Exception {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        try (Writer w = new OutputStreamWriter(bos, StandardCharsets.UTF_8)) {
            CsvWriter.write(df, w, CsvOptions.defaults());
        }
        return bos.toByteArray();
    }

    private static DataFrame decodeCsv(byte[] data) throws Exception {
        Path tmp = Files.createTempFile("minio-df-", ".csv");
        try {
            Files.write(tmp, data);
            return DataFrame.readCsv(tmp.toString());
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    private static byte[] encodeParquet(DataFrame df) throws Exception {
        Path tmp = Files.createTempFile("minio-df-", ".parquet");
        try {
            df.writeParquet(tmp.toString());
            return Files.readAllBytes(tmp);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    private static DataFrame decodeParquet(byte[] data) throws Exception {
        Path tmp = Files.createTempFile("minio-df-", ".parquet");
        try {
            Files.write(tmp, data);
            return DataFrame.readParquet(tmp.toString());
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    private static byte[] encodeBytesFrame(DataFrame df) {
        for (int c = 0; c < df.columnCount(); c++) {
            String name = df.column(c).name();
            for (int r = 0; r < df.rowCount(); r++) {
                Object cell = df.get(r, name);
                byte[] b = cellToBytes(cell);
                if (b != null) return b;
            }
        }
        return encodeJsonl(df);
    }

    private static byte[] cellToBytes(Object cell) {
        if (cell == null) return null;
        if (cell instanceof byte[] b) return b;
        if (cell instanceof BinaryData bd) return bd.getData();
        if (cell instanceof String s) return s.getBytes(StandardCharsets.UTF_8);
        return null;
    }

    static byte[] maybeCompress(byte[] data, MinioOptions.Compression c) throws IOException {
        if (data == null) return new byte[0];
        if (c == null || c == MinioOptions.Compression.NONE) return data;
        return switch (c) {
            case GZIP -> {
                ByteArrayOutputStream bos = new ByteArrayOutputStream(data.length / 2 + 32);
                try (GZIPOutputStream gz = new GZIPOutputStream(bos)) { gz.write(data); }
                yield bos.toByteArray();
            }
            case SNAPPY -> Snappy.compress(data);
            case ZSTD -> {
                long max = Zstd.compressBound(data.length);
                byte[] out = new byte[(int) max];
                long n = Zstd.compress(out, data, 3);
                if (Zstd.isError(n)) throw new IOException("ZSTD compress: " + Zstd.getErrorName(n));
                byte[] exact = new byte[(int) n];
                System.arraycopy(out, 0, exact, 0, (int) n);
                yield exact;
            }
            case NONE -> data;
        };
    }

    static byte[] maybeDecompress(byte[] data, MinioOptions.Compression requested,
                                  MinioOptions.Compression detected) throws IOException {
        if (data == null) return new byte[0];
        MinioOptions.Compression c = requested != null && requested != MinioOptions.Compression.NONE
                ? requested
                : (detected == null ? MinioOptions.Compression.NONE : detected);
        if (c == MinioOptions.Compression.NONE) return data;
        return switch (c) {
            case GZIP -> {
                try (GZIPInputStream gz = new GZIPInputStream(new ByteArrayInputStream(data))) {
                    yield gz.readAllBytes();
                }
            }
            case SNAPPY -> Snappy.uncompress(data);
            case ZSTD -> {
                long size = Zstd.decompressedSize(data);
                if (size <= 0 || size > Integer.MAX_VALUE) size = (long) data.length * 8L;
                byte[] out = new byte[(int) size];
                long n = Zstd.decompress(out, data);
                if (Zstd.isError(n)) throw new IOException("ZSTD decompress: " + Zstd.getErrorName(n));
                if (n == out.length) yield out;
                byte[] exact = new byte[(int) n];
                System.arraycopy(out, 0, exact, 0, (int) n);
                yield exact;
            }
            case NONE -> data;
        };
    }

    private static MinioOptions.Compression detectCompression(MinioOptions o, Headers headers) {
        if (o.compression() != null && o.compression() != MinioOptions.Compression.NONE) {
            return o.compression();
        }
        if (headers != null) {
            String ce = headers.get("Content-Encoding");
            if (ce != null) {
                String l = ce.toLowerCase(Locale.ROOT);
                if (l.contains("gzip")) return MinioOptions.Compression.GZIP;
                if (l.contains("zstd")) return MinioOptions.Compression.ZSTD;
                if (l.contains("snappy")) return MinioOptions.Compression.SNAPPY;
            }
            String ct = headers.get("Content-Type");
            if (ct != null) {
                String l = ct.toLowerCase(Locale.ROOT);
                if (l.contains("gzip")) return MinioOptions.Compression.GZIP;
                if (l.contains("zstd")) return MinioOptions.Compression.ZSTD;
                if (l.contains("snappy")) return MinioOptions.Compression.SNAPPY;
            }
        }
        return guessCompressionFromKey(o.objectKey());
    }

    static MinioOptions.Compression guessCompressionFromKey(String key) {
        if (key == null) return MinioOptions.Compression.NONE;
        String k = key.toLowerCase(Locale.ROOT);
        if (k.endsWith(".gz") || k.endsWith(".gzip")) return MinioOptions.Compression.GZIP;
        if (k.endsWith(".zst") || k.endsWith(".zstd")) return MinioOptions.Compression.ZSTD;
        if (k.endsWith(".snappy") || k.endsWith(".sz")) return MinioOptions.Compression.SNAPPY;
        return MinioOptions.Compression.NONE;
    }

    static MinioOptions.Format guessFormat(String key, byte[] raw, MinioOptions.Format fallback) {
        if (key != null) {
            String k = key.toLowerCase(Locale.ROOT);
            if (k.endsWith(".gz")) k = k.substring(0, k.length() - 3);
            else if (k.endsWith(".zst") || k.endsWith(".zstd")) k = k.replaceAll("\\.zstd?$", "");
            if (k.endsWith(".parquet") || k.endsWith(".pq")) return MinioOptions.Format.PARQUET;
            if (k.endsWith(".jsonl") || k.endsWith(".ndjson")) return MinioOptions.Format.JSONL;
            if (k.endsWith(".json")) return MinioOptions.Format.JSON;
            if (k.endsWith(".csv") || k.endsWith(".tsv")) return MinioOptions.Format.CSV;
            if (k.endsWith(".bin") || k.endsWith(".bytes") || k.endsWith(".dat")) return MinioOptions.Format.BYTES;
        }
        if (raw != null && raw.length >= 4) {
            if (raw[0] == 'P' && raw[1] == 'A' && raw[2] == 'R' && raw[3] == '1') {
                return MinioOptions.Format.PARQUET;
            }
            if (raw[0] == '[') return MinioOptions.Format.JSON;
            if (raw[0] == '{') return MinioOptions.Format.JSONL;
        }
        return fallback == null ? MinioOptions.Format.JSONL : fallback;
    }

    private static io.minio.messages.InputSerialization buildInputSerialization(MinioOptions o) {
        String s = o.selectInputSerialization() == null
                ? "csv" : o.selectInputSerialization().toLowerCase(Locale.ROOT);
        if (s.contains("parquet")) return io.minio.messages.InputSerialization.newParquet();
        if (s.contains("json")) {
            return io.minio.messages.InputSerialization.newJSON(
                    io.minio.messages.InputSerialization.CompressionType.NONE,
                    io.minio.messages.InputSerialization.JsonType.LINES);
        }
        // compression, allowQuotedRecordDelimiter, comments, fieldDelimiter,
        // fileHeaderInfo, quoteCharacter, quoteEscapeCharacter, recordDelimiter
        return io.minio.messages.InputSerialization.newCSV(
                io.minio.messages.InputSerialization.CompressionType.NONE,
                false,
                null,
                null,
                io.minio.messages.InputSerialization.FileHeaderInfo.USE,
                null,
                null,
                null);
    }

    private static io.minio.messages.OutputSerialization buildOutputSerialization(MinioOptions o) {
        String s = o.selectOutputSerialization() == null
                ? "csv" : o.selectOutputSerialization().toLowerCase(Locale.ROOT);
        if (s.contains("json")) {
            return io.minio.messages.OutputSerialization.newJSON(null);
        }
        // fieldDelimiter, quoteCharacter, quoteEscapeCharacter, quoteFields, recordDelimiter
        return io.minio.messages.OutputSerialization.newCSV(null, null, null, null, null);
    }

    private static Http.Method parseMethod(String method) {
        if (method == null || method.isBlank()) return Http.Method.GET;
        return switch (method.trim().toUpperCase(Locale.ROOT)) {
            case "PUT" -> Http.Method.PUT;
            case "POST" -> Http.Method.POST;
            case "DELETE" -> Http.Method.DELETE;
            case "HEAD" -> Http.Method.HEAD;
            default -> Http.Method.GET;
        };
    }

    private static String normalizeEndpoint(String endpoint) {
        if (endpoint == null || endpoint.isBlank()) return "http://127.0.0.1:9000";
        String e = endpoint.trim();
        if (!e.contains("://")) e = "http://" + e;
        while (e.endsWith("/")) e = e.substring(0, e.length() - 1);
        return e;
    }

    private static String firstEnv(String... keys) {
        for (String k : keys) {
            String v = System.getenv(k);
            if (v != null && !v.isBlank()) return v.trim();
        }
        return null;
    }

    /** Thrown when IfExists.SKIP hits an existing object. */
    public static final class SkipWrite extends MinioException {
        public SkipWrite(String message) {
            super(message, null, "ifExists", null, null, "SkipWrite");
        }
    }

    /** Lightweight object listing row. */
    public record MinioObjectInfo(
            String objectName,
            long size,
            String etag,
            String storageClass,
            boolean isDir,
            boolean isLatest,
            String versionId,
            boolean isDeleteMarker,
            java.time.ZonedDateTime lastModified,
            Map<String, String> userMetadata
    ) {
        static MinioObjectInfo from(Item item) {
            Map<String, String> meta = item.userMetadata() == null
                    ? Map.of() : Map.copyOf(item.userMetadata());
            return new MinioObjectInfo(
                    item.objectName(), item.size(), item.etag(), item.storageClass(),
                    item.isDir(), item.isLatest(), item.versionId(), item.isDeleteMarker(),
                    item.lastModified(), meta);
        }
    }
}
