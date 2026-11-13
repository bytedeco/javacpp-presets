import org.bytedeco.javacpp.*;
import org.bytedeco.blosc2.*;
import org.bytedeco.blosc2.global.blosc2;

/**
 * Compresses and decompresses a buffer of 32-bit integers the way Zarr and N5 chunks are encoded with Blosc.
 * Stock C-Blosc2 chunks cannot be decoded by C-Blosc 1.x readers (such as numcodecs), not even those written
 * by blosc1_compress(). blosc2_ctx_set_blosc1_format() (C-Blosc2 3.6.2 and later) makes the context write
 * chunks that C-Blosc 1.x can decode. C-Blosc2 decodes both kinds with blosc1_decompress().
 */
public class BloscRoundTripExample {
    public static void main(String[] args) {
        blosc2.blosc2_init();
        try {
            blosc2.blosc2_set_nthreads((short)4);

            int n = 1 << 20;
            IntPointer src = new IntPointer(n);
            for (int i = 0; i < n; i++) {
                src.put(i, i / 16);
            }
            long nbytes = (long)n * Integer.BYTES;

            // clevel 5, byte shuffle, typesize 4, written in the Blosc1 chunk format
            blosc2_cparams cparams = new blosc2_cparams();
            cparams.compcode((byte)blosc2.BLOSC_ZSTD);
            cparams.clevel((byte)5);
            cparams.typesize(Integer.BYTES);
            cparams.nthreads((short)4);
            cparams.splitmode(blosc2.BLOSC_FORWARD_COMPAT_SPLIT);
            cparams.filters(blosc2.BLOSC2_MAX_FILTERS - 1, (byte)blosc2.BLOSC_SHUFFLE);
            blosc2_context cctx = blosc2.blosc2_create_cctx(cparams);
            if (blosc2.blosc2_ctx_set_blosc1_format(cctx, true) < 0) {
                throw new RuntimeException("Blosc1 format is not available for this setup");
            }
            BytePointer compressed = new BytePointer(nbytes + blosc2.BLOSC2_MAX_OVERHEAD);
            int csize = blosc2.blosc2_compress_ctx(cctx, src, (int)nbytes, compressed, (int)compressed.capacity());
            blosc2.blosc2_free_ctx(cctx);
            if (csize <= 0) {
                throw new RuntimeException("Compression failed: " + csize);
            }
            System.out.println("Compressed " + nbytes + " bytes into " + csize + " bytes");

            // The uncompressed size is stored in the chunk header
            SizeTPointer uncompressedSize = new SizeTPointer(1);
            SizeTPointer compressedSize = new SizeTPointer(1);
            SizeTPointer blockSize = new SizeTPointer(1);
            blosc2.blosc1_cbuffer_sizes(compressed, uncompressedSize, compressedSize, blockSize);

            IntPointer dest = new IntPointer(n);
            int dsize = blosc2.blosc1_decompress(compressed, dest, uncompressedSize.get());
            if (dsize != nbytes) {
                throw new RuntimeException("Decompression failed: " + dsize);
            }
            for (int i = 0; i < n; i++) {
                if (src.get(i) != dest.get(i)) {
                    throw new RuntimeException("Mismatch at index " + i);
                }
            }
            System.out.println("Round trip OK");
        } finally {
            blosc2.blosc2_destroy();
        }
    }
}
