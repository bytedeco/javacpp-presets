import java.nio.ByteBuffer;
import org.bytedeco.javacpp.*;
import org.bytedeco.lz4.*;
import org.bytedeco.lz4.global.lz4;

public final class LZ4FrameCompressionExample {

    private static final int NUM_VALUES = 10 * 1024 * 1024; // 10MB

    public static void main(String[] args) throws LZ4Exception {
        // Print LZ4 version
        System.out.println("LZ4 Version: " + lz4.LZ4_VERSION_STRING.getString());

        // Generate some data
        final ByteBuffer data = ByteBuffer.allocateDirect(NUM_VALUES);
        for (int i = 0; i < NUM_VALUES; i++) {
            data.put((byte) i);
        }
        data.position(0);

        // Compress
        final ByteBuffer compressed = compress(data);
        System.out.println("Uncompressed size: " + data.limit());
        System.out.println("Compressed size: " + compressed.limit());

        // Decompress
        final ByteBuffer decompressed = decompress(compressed, data.limit());

        // Verify that decompressed == data
        for (int i = 0; i < NUM_VALUES; i++) {
            if (data.get(i) != decompressed.get(i)) {
                throw new IllegalStateException("Input and output differ.");
            }
        }
        System.out.println("Verified that input data == output data");
    }

    private static ByteBuffer compress(ByteBuffer data) {
        // Output buffer
        final int maxCompressedSize = (int) lz4.LZ4F_compressFrameBound(data.limit(), null);
        final ByteBuffer compressed = ByteBuffer.allocateDirect(maxCompressedSize);

        final Pointer dataPointer = new Pointer(data);
        final Pointer dstPointer = new Pointer(compressed);
        final long compressedSize = lz4.LZ4F_compressFrame(dstPointer, compressed.limit(), dataPointer, data.limit(),
                null);
        compressed.limit((int) compressedSize);
        return compressed;
    }

    private static ByteBuffer decompress(ByteBuffer compressed, int uncompressedSize) throws LZ4Exception {
        final LZ4FDecompressionContext dctx = new LZ4FDecompressionContext();
        final long ctxError = lz4.LZ4F_createDecompressionContext(dctx, lz4.LZ4F_VERSION);
        checkForError(ctxError);

        // Output buffer
        final ByteBuffer decompressed = ByteBuffer.allocateDirect(uncompressedSize);

        final SizeTPointer dstSize = new SizeTPointer(1);
        final SizeTPointer srcSize = new SizeTPointer(1);

        try {
            long ret;
            do {
                dstSize.put(decompressed.remaining());
                srcSize.put(compressed.limit());
                final Pointer dstPointer = new Pointer(decompressed);
                final Pointer compressedPointer = new Pointer(compressed);

                ret = lz4.LZ4F_decompress(dctx, dstPointer, dstSize, compressedPointer, srcSize, null);
                checkForError(ret);
                decompressed.position(decompressed.position() + (int) dstSize.get());
                compressed.position(compressed.position() + (int) srcSize.get());
            } while (ret != 0);

        } finally {
            lz4.LZ4F_freeDecompressionContext(dctx);
        }

        decompressed.position(0);
        return decompressed;
    }

    private static void checkForError(long errorCode) throws LZ4Exception {
        if (lz4.LZ4F_isError(errorCode) != 0) {
            throw new LZ4Exception(lz4.LZ4F_getErrorName(errorCode).getString());
        }
    }

    private static final class LZ4Exception extends Exception {
        public LZ4Exception(final String message) {
            super(message);
        }
    }
}
