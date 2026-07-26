package org.bytedeco.pytorch.data.parquet;

import java.io.IOException;
import java.nio.ByteBuffer;

import com.github.luben.zstd.Zstd;
import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.compression.CompressionCodecFactory;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.xerial.snappy.Snappy;

/**
 * Pure-Java CompressionCodecFactory (no Hadoop).
 * Supports ZSTD (via zstd-jni) and SNAPPY (via snappy-java) decompression.
 * Write/compress throws {@link UnsupportedOperationException}.
 */
public final class ZstdCodecFactory implements CompressionCodecFactory {

    /** Singleton instance for reuse across readers/writers. */
    public static final ZstdCodecFactory INSTANCE = new ZstdCodecFactory();

    private ZstdCodecFactory() {}

    @Override
    public CompressionCodecFactory.BytesInputCompressor getCompressor(CompressionCodecName codecName) {
        throw new UnsupportedOperationException(
            "Compression not supported in pure-Java codec factory: " + codecName);
    }

    @Override
    public CompressionCodecFactory.BytesInputDecompressor getDecompressor(CompressionCodecName codecName) {
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

    public static final class NoopDecompressor
            implements CompressionCodecFactory.BytesInputDecompressor {
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

    public static final class ZstdDecompressor
            implements CompressionCodecFactory.BytesInputDecompressor {
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

    public static final class SnappyDecompressor
            implements CompressionCodecFactory.BytesInputDecompressor {
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
            output.position(output.position() + outputLen);
        }
        @Override public void release() {}
    }
}
