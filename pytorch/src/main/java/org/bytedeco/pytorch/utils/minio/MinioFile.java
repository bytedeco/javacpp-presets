package org.bytedeco.pytorch.utils.minio;

import io.minio.ObjectWriteResponse;
import io.minio.StatObjectResponse;
import org.bytedeco.pytorch.dataframe.dtype.BinaryData;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Object-level binary file abstraction over MinIO / S3.
 *
 * <pre>{@code
 * MinioFile f = MinioFile.of(m, "datasets", "train/img-0.bin");
 * f.write(bytes);
 * byte[] back = f.readAllBytes();
 * byte[] slice = f.readRange(0, 1024);
 * f.copyTo(MinioFile.of(m, "datasets", "train/img-0-copy.bin"));
 * f.delete();
 * }</pre>
 */
public final class MinioFile {

    private final Minio minio;
    private final String bucket;
    private final String objectKey;
    private final String versionId;

    private MinioFile(Minio minio, String bucket, String objectKey, String versionId) {
        this.minio = Objects.requireNonNull(minio, "minio");
        this.bucket = Objects.requireNonNull(bucket, "bucket");
        this.objectKey = Objects.requireNonNull(objectKey, "objectKey");
        this.versionId = versionId;
    }

    public static MinioFile of(Minio minio, String bucket, String objectKey) {
        return new MinioFile(minio, bucket, objectKey, null);
    }

    public static MinioFile of(Minio minio, String bucket, String objectKey, String versionId) {
        return new MinioFile(minio, bucket, objectKey, versionId);
    }

    public Minio minio() {
        return minio;
    }

    public String bucket() {
        return bucket;
    }

    public String objectKey() {
        return objectKey;
    }

    public String versionId() {
        return versionId;
    }

    public MinioOptions baseOptions() {
        MinioOptions.Builder b = MinioOptions.builder()
                .bucket(bucket)
                .objectKey(objectKey)
                .format(MinioOptions.Format.BYTES)
                .contentType("application/octet-stream")
                .ensureBucket(true);
        if (versionId != null) b.versionId(versionId);
        return b.build();
    }

    /** Read entire object. */
    public byte[] readAllBytes() {
        return minio.getBytes(baseOptions());
    }

    /** Read {@code [offset, offset+length)} range. */
    public byte[] readRange(long offset, long length) {
        return minio.getRange(bucket, objectKey, offset, length);
    }

    /** Write full object (replace). */
    public ObjectWriteResponse write(byte[] data) {
        return minio.putBytes(data, baseOptions());
    }

    /**
     * Write at logical offset. S3/MinIO objects are immutable — when {@code offset == 0}
     * this is a full replace; non-zero offset appends via {@code AppendObject} when supported,
     * otherwise compose(prefix + new) is attempted through raw client.
     */
    public ObjectWriteResponse write(byte[] data, long offset) {
        Objects.requireNonNull(data, "data");
        if (offset <= 0) return write(data);
        try {
            // Prefer appendObject API (MinIO 9.x)
            io.minio.ObjectWriteResponse resp = minio.raw().appendObject(
                    io.minio.AppendObjectArgs.builder()
                            .bucket(bucket)
                            .object(objectKey)
                            .stream(new ByteArrayInputStream(data), (long) data.length)
                            .build());
            minio.metrics().recordPut(data.length, true);
            return resp;
        } catch (Exception e) {
            // Fallback: read-modify-write
            byte[] existing;
            try {
                existing = minio.objectExists(bucket, objectKey) ? readAllBytes() : new byte[0];
            } catch (Exception ex) {
                existing = new byte[0];
            }
            long need = offset + data.length;
            if (need > Integer.MAX_VALUE) {
                throw new MinioException("object too large for RMW fallback", e, "write", bucket, objectKey);
            }
            byte[] merged = new byte[(int) Math.max(existing.length, need)];
            System.arraycopy(existing, 0, merged, 0, existing.length);
            System.arraycopy(data, 0, merged, (int) offset, data.length);
            return write(merged);
        }
    }

    public ObjectWriteResponse write(BinaryData binary) {
        Objects.requireNonNull(binary, "binary");
        return minio.putBinaryData(binary, baseOptions().toBuilder()
                .objectKey(objectKey)
                .bucket(bucket)
                .build());
    }

    public ObjectWriteResponse write(InputStream stream, long size) {
        return minio.putStream(stream, size, baseOptions());
    }

    public ObjectWriteResponse upload(Path path) {
        return minio.uploadFile(path, baseOptions());
    }

    public void download(Path path) {
        minio.downloadObject(path, baseOptions());
    }

    /** Server-side copy to another MinioFile. */
    public ObjectWriteResponse copyTo(MinioFile target) {
        Objects.requireNonNull(target, "target");
        return minio.copyObject(bucket, objectKey, target.bucket, target.objectKey);
    }

    public void delete() {
        minio.removeObject(baseOptions());
    }

    public void deleteVersion(String versionId) {
        minio.removeObjectVersion(bucket, objectKey, versionId);
    }

    public boolean exists() {
        return minio.objectExists(bucket, objectKey);
    }

    public StatObjectResponse stat() {
        return minio.statObject(baseOptions());
    }

    /** Metadata map: size, etag, contentType, lastModified, versionId, user meta. */
    public Map<String, Object> getMetadata() {
        StatObjectResponse st = stat();
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("bucket", bucket);
        m.put("objectKey", objectKey);
        m.put("size", st.size());
        m.put("etag", st.etag());
        m.put("contentType", st.contentType());
        m.put("lastModified", st.lastModified());
        m.put("versionId", st.versionId());
        m.put("deleteMarker", st.deleteMarker());
        if (st.userMetadata() != null) {
            m.put("userMetadata", st.userMetadata());
        }
        return m;
    }

    public BinaryData toBinaryData() {
        return new BinaryData(objectKey, readAllBytes());
    }

    public MinioFile withVersion(String versionId) {
        return new MinioFile(minio, bucket, objectKey, versionId);
    }

    @Override
    public String toString() {
        return "MinioFile{s3://" + bucket + "/" + objectKey
                + (versionId != null ? "?versionId=" + versionId : "")
                + "}";
    }
}
