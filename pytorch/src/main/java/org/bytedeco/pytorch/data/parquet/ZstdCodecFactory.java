package org.bytedeco.pytorch.data.parquet;

import java.io.IOException;
import java.nio.ByteBuffer;

import com.github.luben.zstd.Zstd;
import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.compression.CompressionCodecFactory;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.xerial.snappy.Snappy;

/**
 * Pure-Java {@link CompressionCodecFactory} (no Hadoop codecs).
 * Supports UNCOMPRESSED / ZSTD (zstd-jni) / SNAPPY (snappy-java) for both
 * compress and decompress so {@link LocalParquetWriter} can drop
 * {@code hadoop-client-runtime}.
 */
public final class ZstdCodecFactory implements CompressionCodecFactory {

    /** Singleton instance for reuse across readers/writers. */
    public static final ZstdCodecFactory INSTANCE = new ZstdCodecFactory();

    private ZstdCodecFactory() {}

    @Override
    public BytesInputCompressor getCompressor(CompressionCodecName codecName) {
        switch (codecName) {
            case UNCOMPRESSED: return NoopCompressor.INSTANCE;
            case ZSTD:         return ZstdCompressor.INSTANCE;
            case SNAPPY:       return SnappyCompressor.INSTANCE;
            default:
                throw new UnsupportedOperationException(
                    "Unsupported codec for pure-Java compress (no Hadoop): " + codecName);
        }
    }

    @Override
    public BytesInputDecompressor getDecompressor(CompressionCodecName codecName) {
        switch (codecName) {
            case UNCOMPRESSED: return NoopDecompressor.INSTANCE;
            case ZSTD:         return ZstdDecompressor.INSTANCE;
            case SNAPPY:       return SnappyDecompressor.INSTANCE;
            default:
                throw new UnsupportedOperationException(
                    "Unsupported codec (no Hadoop): " + codecName);
        }
    }

    @Override
    public void release() {}

    // ---- Noop ----------------------------------------------------------------

    public static final class NoopCompressor implements BytesInputCompressor {
        static final NoopCompressor INSTANCE = new NoopCompressor();
        private NoopCompressor() {}
        @Override public BytesInput compress(BytesInput bytes) { return bytes; }
        @Override public CompressionCodecName getCodecName() { return CompressionCodecName.UNCOMPRESSED; }
        @Override public void release() {}
    }

    public static final class NoopDecompressor implements BytesInputDecompressor {
        static final NoopDecompressor INSTANCE = new NoopDecompressor();
        private NoopDecompressor() {}
        @Override public BytesInput decompress(BytesInput bytes, int uncompressedSize) { return bytes; }
        @Override public void decompress(ByteBuffer input, int inputLen,
                                        ByteBuffer output, int outputLen) {
            int saved = input.limit();
            input.limit(input.position() + inputLen);
            output.put(input);
            input.limit(saved);
        }
        @Override public void release() {}
    }

    // ---- ZSTD ----------------------------------------------------------------

    public static final class ZstdCompressor implements BytesInputCompressor {
        static final ZstdCompressor INSTANCE = new ZstdCompressor();
        private ZstdCompressor() {}
        @Override public BytesInput compress(BytesInput bytes) throws IOException {
            byte[] in = bytes.toByteArray();
            long max = Zstd.compressBound(in.length);
            byte[] out = new byte[(int) max];
            // zstd-jni: compress(dst, src, level)
            long n = Zstd.compress(out, in, 3);
            if (Zstd.isError(n)) throw new IOException("ZSTD compress error: " + Zstd.getErrorName(n));
            return BytesInput.from(out, 0, (int) n);
        }
        @Override public CompressionCodecName getCodecName() { return CompressionCodecName.ZSTD; }
        @Override public void release() {}
    }

    public static final class ZstdDecompressor implements BytesInputDecompressor {
        static final ZstdDecompressor INSTANCE = new ZstdDecompressor();
        private ZstdDecompressor() {}
        @Override public BytesInput decompress(BytesInput bytes, int uncompressedSize) throws IOException {
            byte[] out = new byte[uncompressedSize];
            long n = Zstd.decompress(out, bytes.toByteArray());
            if (n != uncompressedSize)
                throw new IOException("ZSTD size mismatch: expected " + uncompressedSize + " got " + n);
            return BytesInput.from(out);
        }
        @Override public void decompress(ByteBuffer input, int inputLen,
                                        ByteBuffer output, int outputLen) throws IOException {
            ByteBuffer in = input.duplicate();
            in.limit(in.position() + inputLen);
            @SuppressWarnings("deprecation")
            long n = Zstd.decompress(output, in);
            if (n != outputLen)
                throw new IOException("ZSTD size mismatch: expected " + outputLen + " got " + n);
            output.position(output.position() + outputLen);
        }
        @Override public void release() {}
    }

    // ---- SNAPPY --------------------------------------------------------------

    public static final class SnappyCompressor implements BytesInputCompressor {
        static final SnappyCompressor INSTANCE = new SnappyCompressor();
        private SnappyCompressor() {}
        @Override public BytesInput compress(BytesInput bytes) throws IOException {
            byte[] in = bytes.toByteArray();
            byte[] out = Snappy.compress(in);
            return BytesInput.from(out);
        }
        @Override public CompressionCodecName getCodecName() { return CompressionCodecName.SNAPPY; }
        @Override public void release() {}
    }

    public static final class SnappyDecompressor implements BytesInputDecompressor {
        static final SnappyDecompressor INSTANCE = new SnappyDecompressor();
        private SnappyDecompressor() {}
        @Override public BytesInput decompress(BytesInput bytes, int uncompressedSize) throws IOException {
            byte[] out = new byte[uncompressedSize];
            long n = Snappy.uncompress(bytes.toByteArray(), 0, (int) bytes.size(), out, 0);
            if (n != uncompressedSize)
                throw new IOException("SNAPPY size mismatch: expected " + uncompressedSize + " got " + n);
            return BytesInput.from(out);
        }
        @Override public void decompress(ByteBuffer input, int inputLen,
                                        ByteBuffer output, int outputLen) throws IOException {
            byte[] inArr;
            if (input.hasArray()) {
                int off = input.arrayOffset() + input.position();
                inArr = java.util.Arrays.copyOfRange(input.array(), off, off + inputLen);
            } else {
                inArr = new byte[inputLen];
                ByteBuffer dup = input.duplicate();
                dup.get(inArr);
            }
            byte[] outArr = new byte[outputLen];
            Snappy.uncompress(inArr, 0, inArr.length, outArr, 0);
            output.put(outArr);
        }
        @Override public void release() {}
    }
}
