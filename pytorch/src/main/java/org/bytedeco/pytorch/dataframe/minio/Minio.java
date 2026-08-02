package org.bytedeco.pytorch.dataframe.minio;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.BinaryData;
import org.bytedeco.pytorch.utils.minio.MinioException;
import org.bytedeco.pytorch.utils.minio.MinioOptions;

import java.util.Objects;

/**
 * DataFrame-facing MinIO helpers — thin façade over
 * {@link org.bytedeco.pytorch.utils.minio.Minio}.
 *
 * <pre>{@code
 * try (var m = org.bytedeco.pytorch.utils.minio.Minio.connect(
 *         "http://127.0.0.1:9000", "minioadmin", "minioadmin")) {
 *     int n = Minio.toMinio(df, m, opts);
 *     DataFrame back = Minio.readMinio(m, opts);
 * }
 * }</pre>
 *
 * <p>Prefer {@link DataFrame#toMinio} / {@link DataFrame#readMinio} entry points.
 */
public final class Minio {

    private Minio() {}

    public static int toMinio(DataFrame df, org.bytedeco.pytorch.utils.minio.Minio client, MinioOptions options) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(client, "client");
        return client.writeDataFrame(df, options);
    }

    public static int toMinio(DataFrame df, org.bytedeco.pytorch.utils.minio.Minio client,
                              String bucket, String objectKey) {
        return toMinio(df, client, MinioOptions.builder().bucket(bucket).objectKey(objectKey).build());
    }

    public static int toMinio(DataFrame df, org.bytedeco.pytorch.utils.minio.Minio client,
                              String bucket, String objectKey, String contentType) {
        return toMinio(df, client, MinioOptions.builder()
                .bucket(bucket)
                .objectKey(objectKey)
                .contentType(contentType)
                .build());
    }

    public static DataFrame readMinio(org.bytedeco.pytorch.utils.minio.Minio client, MinioOptions options) {
        Objects.requireNonNull(client, "client");
        return client.readDataFrame(options);
    }

    public static DataFrame readMinio(org.bytedeco.pytorch.utils.minio.Minio client,
                                      String bucket, String objectKey) {
        return readMinio(client, MinioOptions.builder().bucket(bucket).objectKey(objectKey).build());
    }

    public static int toMinio(DataFrame df, String uri, MinioOptions options) {
        try (org.bytedeco.pytorch.utils.minio.Minio m = org.bytedeco.pytorch.utils.minio.Minio.connectUri(uri)) {
            MinioOptions o = options;
            if (o == null) o = MinioOptions.fromUri(uri);
            else {
                // merge path components from uri when missing
                MinioOptions fromUri = MinioOptions.fromUri(uri);
                MinioOptions.Builder b = o.toBuilder();
                if (o.bucket() == null) b.bucket(fromUri.bucket());
                if (o.objectKey() == null) b.objectKey(fromUri.objectKey());
                o = b.build();
            }
            return m.writeDataFrame(df, o);
        }
    }

    public static DataFrame readMinio(String uri, MinioOptions options) {
        try (org.bytedeco.pytorch.utils.minio.Minio m = org.bytedeco.pytorch.utils.minio.Minio.connectUri(uri)) {
            MinioOptions o = options == null ? MinioOptions.fromUri(uri) : options;
            if (o.bucket() == null || o.objectKey() == null) {
                MinioOptions fromUri = MinioOptions.fromUri(uri);
                o = o.toBuilder()
                        .bucket(o.bucket() != null ? o.bucket() : fromUri.bucket())
                        .objectKey(o.objectKey() != null ? o.objectKey() : fromUri.objectKey())
                        .build();
            }
            return m.readDataFrame(o);
        }
    }

    public static int writeBinary(DataFrame df, String column,
                                  org.bytedeco.pytorch.utils.minio.Minio client, MinioOptions options) {
        return client.writeBinaryColumn(df, column, options);
    }

    public static int putBinary(org.bytedeco.pytorch.utils.minio.Minio client, BinaryData data, MinioOptions options) {
        if (data == null) throw new MinioException("binary data required", null, "putBinary", null, null);
        client.putBinaryData(data, options);
        return data.getData() == null ? 0 : data.getData().length;
    }
}
