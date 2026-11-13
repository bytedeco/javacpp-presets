import org.bytedeco.javacpp.*;
import org.bytedeco.blosc2.*;
import org.bytedeco.blosc2.global.blosc2;
import java.nio.file.*;

public class BloscChunkWriter {
    static byte gen(long i) { return (byte)((i / 4) * 7 + (i % 4)); }
    public static void main(String[] a) throws Exception {
        String mode = a[0]; Path out = Paths.get(a[1]); Files.createDirectories(out);
        String[] codecs = {"blosclz", "lz4", "lz4hc", "zlib", "zstd"};
        int[] tss = {1, 2, 3, 4, 8, 12, 16, 17};
        int[] sizes = {1, 100, 1000, 4097, 100003, 1 << 20};
        int fails = 0, n = 0;
        blosc2.blosc2_init();
        for (int ci = 0; ci < codecs.length; ci++)
         for (int f = 0; f <= 2; f++)
          for (int ts : tss)
           for (int size : sizes) {
            BytePointer src = new BytePointer(size);
            for (int i = 0; i < size; i++) src.put(i, gen(i));
            BytePointer dst = new BytePointer(size + blosc2.BLOSC2_MAX_OVERHEAD);
            int cs;
            if (mode.equals("ctx")) {
                blosc2_cparams p = new blosc2_cparams();
                p.compcode((byte)blosc2.blosc2_compname_to_compcode(codecs[ci]));
                p.clevel((byte)5); p.typesize(ts); p.nthreads((short)1);
                p.splitmode(blosc2.BLOSC_FORWARD_COMPAT_SPLIT);
                p.filters(blosc2.BLOSC2_MAX_FILTERS - 1, (byte)f);
                blosc2_context ctx = blosc2.blosc2_create_cctx(p);
                int r = blosc2.blosc2_ctx_set_blosc1_format(ctx, true);
                if (r < 0) throw new RuntimeException("set_blosc1_format " + r);
                cs = blosc2.blosc2_compress_ctx(ctx, src, size, dst, size + blosc2.BLOSC2_MAX_OVERHEAD);
                blosc2.blosc2_free_ctx(ctx);
            } else {
                blosc2.blosc1_set_compressor(codecs[ci]);
                cs = blosc2.blosc1_compress(5, f, ts, size, src, dst, size + blosc2.BLOSC2_MAX_OVERHEAD);
            }
            n++;
            if (cs <= 0) { System.out.println("compress fail " + codecs[ci] + " f" + f + " ts" + ts + " n" + size + " -> " + cs); fails++; continue; }
            byte[] b = new byte[cs]; dst.position(0).get(b, 0, cs);
            Files.write(out.resolve(codecs[ci] + "_f" + f + "_t" + ts + "_n" + size + ".bin"), b);
        }
        blosc2.blosc2_destroy();
        System.out.println(mode + ": wrote " + (n - fails) + "/" + n + " chunks");
    }
}
