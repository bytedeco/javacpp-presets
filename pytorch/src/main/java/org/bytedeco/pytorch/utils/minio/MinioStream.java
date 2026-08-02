package org.bytedeco.pytorch.utils.minio;

import io.minio.GetObjectArgs;
import io.minio.GetObjectResponse;
import io.minio.ObjectWriteResponse;
import io.minio.PutObjectArgs;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.FilterInputStream;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Objects;

/**
 * Stream wrappers for MinIO upload / download.
 *
 * <pre>{@code
 * // upload via OutputStream (buffers then putObject on close)
 * try (OutputStream out = MinioStream.upload(m, opts)) {
 *     out.write(payload);
 * }
 *
 * // download via InputStream
 * try (InputStream in = MinioStream.download(m, opts)) {
 *     byte[] all = in.readAllBytes();
 * }
 * }</pre>
 *
 * <p>For large uploads prefer {@link Minio#putStream} / {@link Minio#uploadFile} directly
 * (true multipart). This class is convenient for moderate in-memory / pipe workloads.
 */
public final class MinioStream implements Closeable {

    private final Minio minio;
    private final MinioOptions options;
    private InputStream download;
    private UploadOutputStream upload;
    private boolean closed;

    private MinioStream(Minio minio, MinioOptions options) {
        this.minio = Objects.requireNonNull(minio, "minio");
        this.options = options == null ? MinioOptions.defaults() : options;
    }

    public static MinioStream open(Minio minio, MinioOptions options) {
        return new MinioStream(minio, options);
    }

    /** Open a download InputStream for the configured object. */
    public static InputStream download(Minio minio, MinioOptions options) {
        Objects.requireNonNull(minio, "minio");
        MinioOptions o = options == null ? minio.options() : options;
        if (o.bucket() == null || o.objectKey() == null) {
            throw new MinioException("bucket/objectKey required", null, "download", null, null);
        }
        minio.ensureOpen();
        try {
            GetObjectArgs.Builder b = GetObjectArgs.builder()
                    .bucket(o.bucket())
                    .object(o.objectKey());
            if (o.versionId() != null) b.versionId(o.versionId());
            if (o.length() != null && o.length() > 0) {
                b.offset(o.offset()).length(o.length());
            } else if (o.offset() > 0) {
                b.offset(o.offset());
            }
            GetObjectResponse resp = minio.raw().getObject(b.build());
            return new FilterInputStream(resp) {
                @Override
                public void close() throws IOException {
                    try {
                        super.close();
                    } finally {
                        try { resp.close(); } catch (Exception ignored) {}
                    }
                }
            };
        } catch (Exception e) {
            throw MinioException.wrap("download", o.bucket(), o.objectKey(), e);
        }
    }

    /**
     * Open an OutputStream that buffers bytes and uploads on {@link OutputStream#close()}.
     * Suitable for moderate payloads; for multi-GB use {@link Minio#putStream}.
     */
    public static OutputStream upload(Minio minio, MinioOptions options) {
        Objects.requireNonNull(minio, "minio");
        MinioOptions o = options == null ? minio.options() : options;
        if (o.bucket() == null || o.objectKey() == null) {
            throw new MinioException("bucket/objectKey required", null, "upload", null, null);
        }
        return new UploadOutputStream(minio, o);
    }

    /** Instance: open download stream (lazy). */
    public InputStream openDownload() {
        if (download == null) download = download(minio, options);
        return download;
    }

    /** Instance: open upload stream (lazy). */
    public OutputStream openUpload() {
        if (upload == null) upload = new UploadOutputStream(minio, options);
        return upload;
    }

    public Minio minio() {
        return minio;
    }

    public MinioOptions options() {
        return options;
    }

    /** Read all download bytes (convenience). */
    public byte[] readAllBytes() throws IOException {
        try (InputStream in = openDownload()) {
            return in.readAllBytes();
        }
    }

    /** Write all bytes via upload stream. */
    public ObjectWriteResponse writeAll(byte[] data) throws IOException {
        Objects.requireNonNull(data, "data");
        try (UploadOutputStream out = new UploadOutputStream(minio, options)) {
            out.write(data);
            out.close();
            return out.response();
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (download != null) {
            try { download.close(); } catch (Exception ignored) {}
            download = null;
        }
        if (upload != null) {
            try { upload.close(); } catch (Exception ignored) {}
            upload = null;
        }
    }

    /** Buffering output stream that puts the object on close. */
    public static final class UploadOutputStream extends FilterOutputStream {
        private final Minio minio;
        private final MinioOptions options;
        private final ByteArrayOutputStream buf;
        private ObjectWriteResponse response;
        private boolean closed;

        UploadOutputStream(Minio minio, MinioOptions options) {
            super(new ByteArrayOutputStream());
            this.minio = minio;
            this.options = options;
            this.buf = (ByteArrayOutputStream) this.out;
        }

        @Override
        public void write(int b) {
            buf.write(b);
        }

        @Override
        public void write(byte[] b, int off, int len) {
            buf.write(b, off, len);
        }

        @Override
        public void close() throws IOException {
            if (closed) return;
            closed = true;
            byte[] data = buf.toByteArray();
            try {
                response = minio.putBytes(data, options.toBuilder()
                        .format(options.format() == null ? MinioOptions.Format.BYTES : options.format())
                        .build());
            } catch (MinioException e) {
                throw new IOException(e.getMessage(), e);
            } catch (Exception e) {
                throw new IOException("upload close failed: " + e.getMessage(), e);
            }
        }

        public ObjectWriteResponse response() {
            return response;
        }

        public long bufferedSize() {
            return buf.size();
        }
    }

    /**
     * Direct put of an InputStream without full buffering (multipart-capable via SDK).
     */
    public static ObjectWriteResponse put(Minio minio, InputStream stream, long objectSize, MinioOptions options) {
        return minio.putStream(stream, objectSize, options);
    }

    /**
     * Direct put of known bytes via stream API.
     */
    public static ObjectWriteResponse put(Minio minio, byte[] data, MinioOptions options) {
        Objects.requireNonNull(data, "data");
        return minio.putStream(new ByteArrayInputStream(data), data.length, options);
    }
}
