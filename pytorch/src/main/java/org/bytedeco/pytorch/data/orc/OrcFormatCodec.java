package org.bytedeco.pytorch.data.orc;

import org.apache.orc.OrcProto;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.Deflater;
import java.util.zip.Inflater;

/**
 * Pure-Java ORC format helpers on top of {@code orc-format} ({@link OrcProto}).
 *
 * <p>No Hadoop, no {@code orc-core}, no Hive vector batches. Handles file tail
 * (postscript / footer / metadata), compression framing (NONE + ZLIB), and
 * integer / bit RLE used by the pure-Java reader and writer.
 *
 * <p>See <a href="https://orc.apache.org/specification/ORCv1/">ORC v1 spec</a>.
 */
public final class OrcFormatCodec {
    public static final byte[] MAGIC = new byte[]{'O', 'R', 'C'};
    public static final String MAGIC_STR = "ORC";
    /** Default compression block size (ORC common default). */
    public static final int DEFAULT_COMPRESSION_BLOCK = 256 * 1024;
    /** ORC 0.12 major/minor version pair written into PostScript. */
    public static final int VERSION_MAJOR = 0;
    public static final int VERSION_MINOR = 12;
    /** Writer id: custom pure-Java (avoid colliding with ORC_JAVA=0). */
    public static final int WRITER_ID = 99;
    public static final int WRITER_VERSION = 9;
    public static final String SOFTWARE_VERSION = "bytedeco-orc-format";

    private OrcFormatCodec() {}

    // ------------------------------------------------------------------ file tail

    /** Parsed ORC file tail (postscript + footer + optional metadata). */
    public static final class FileTail {
        public final OrcProto.PostScript postScript;
        public final OrcProto.Footer footer;
        public final OrcProto.Metadata metadata;
        public final long fileLength;

        public FileTail(OrcProto.PostScript postScript, OrcProto.Footer footer,
                        OrcProto.Metadata metadata, long fileLength) {
            this.postScript = postScript;
            this.footer = footer;
            this.metadata = metadata;
            this.fileLength = fileLength;
        }
    }

    public static FileTail readFileTail(Path path) throws IOException {
        byte[] all = Files.readAllBytes(path);
        return readFileTail(all);
    }

    public static FileTail readFileTail(byte[] all) throws IOException {
        if (all == null || all.length < 4) {
            throw new IOException("ORC file too small");
        }
        // Magic at start
        if (all[0] != 'O' || all[1] != 'R' || all[2] != 'C') {
            throw new IOException("Not an ORC file (missing magic)");
        }
        int psLen = all[all.length - 1] & 0xff;
        if (psLen == 0 || psLen + 1 > all.length) {
            throw new IOException("Invalid ORC postscript length: " + psLen);
        }
        int psStart = all.length - 1 - psLen;
        OrcProto.PostScript ps = OrcProto.PostScript.parseFrom(
            Arrays.copyOfRange(all, psStart, psStart + psLen));
        if (ps.hasMagic() && !MAGIC_STR.equals(ps.getMagic())) {
            throw new IOException("Bad postscript magic: " + ps.getMagic());
        }
        long footerLen = ps.getFooterLength();
        long metaLen = ps.hasMetadataLength() ? ps.getMetadataLength() : 0L;
        int footerEnd = psStart;
        int footerStart = (int) (footerEnd - footerLen);
        int metaStart = (int) (footerStart - metaLen);
        if (metaStart < 3 || footerStart < 3) {
            throw new IOException("ORC tail lengths overrun file");
        }
        OrcProto.CompressionKind kind = ps.hasCompression()
            ? ps.getCompression() : OrcProto.CompressionKind.NONE;
        int blockSize = ps.hasCompressionBlockSize()
            ? (int) ps.getCompressionBlockSize() : DEFAULT_COMPRESSION_BLOCK;

        byte[] footerBytes = decodeCompressed(all, footerStart, (int) footerLen, kind, blockSize);
        OrcProto.Footer footer = OrcProto.Footer.parseFrom(footerBytes);

        OrcProto.Metadata metadata = OrcProto.Metadata.getDefaultInstance();
        if (metaLen > 0) {
            byte[] metaBytes = decodeCompressed(all, metaStart, (int) metaLen, kind, blockSize);
            metadata = OrcProto.Metadata.parseFrom(metaBytes);
        }
        return new FileTail(ps, footer, metadata, all.length);
    }

    /**
     * Write postscript + trailing length byte. Footer/metadata must already have
     * been written (optionally compressed) immediately before the call site.
     */
    public static byte[] buildPostScript(long footerLength, long metadataLength,
                                         OrcProto.CompressionKind compression,
                                         int compressionBlockSize) {
        OrcProto.PostScript.Builder b = OrcProto.PostScript.newBuilder()
            .setFooterLength(footerLength)
            .setCompression(compression == null ? OrcProto.CompressionKind.NONE : compression)
            .setCompressionBlockSize(compressionBlockSize > 0
                ? compressionBlockSize : DEFAULT_COMPRESSION_BLOCK)
            .addVersion(VERSION_MAJOR)
            .addVersion(VERSION_MINOR)
            .setMetadataLength(metadataLength)
            .setWriterVersion(WRITER_VERSION)
            .setMagic(MAGIC_STR);
        return b.build().toByteArray();
    }

    // ------------------------------------------------------------------ compression

    public static OrcProto.CompressionKind toProtoCompress(OrcOptions.Compress c) {
        if (c == null) return OrcProto.CompressionKind.NONE;
        switch (c) {
            case ZLIB: return OrcProto.CompressionKind.ZLIB;
            case NONE: return OrcProto.CompressionKind.NONE;
            case SNAPPY:
            case LZ4:
            case ZSTD:
                // MVP: soft-fallback to ZLIB (no aircompressor / native codecs).
                return OrcProto.CompressionKind.ZLIB;
            default: return OrcProto.CompressionKind.NONE;
        }
    }

    /**
     * Encode a protobuf (or stream body) into ORC compression framing.
     * NONE → raw bytes; ZLIB → 3-byte LE chunk headers + deflated (or original) bodies.
     */
    public static byte[] encodeCompressed(byte[] raw, OrcProto.CompressionKind kind,
                                          int blockSize) throws IOException {
        if (raw == null) return new byte[0];
        if (kind == null || kind == OrcProto.CompressionKind.NONE) {
            return raw;
        }
        if (kind != OrcProto.CompressionKind.ZLIB) {
            throw new IOException("Unsupported ORC compression for pure-Java path: " + kind
                + " (MVP supports NONE and ZLIB)");
        }
        if (blockSize <= 0) blockSize = DEFAULT_COMPRESSION_BLOCK;
        ByteArrayOutputStream out = new ByteArrayOutputStream(raw.length / 2 + 16);
        int off = 0;
        while (off < raw.length) {
            int n = Math.min(blockSize, raw.length - off);
            byte[] chunk = Arrays.copyOfRange(raw, off, off + n);
            byte[] deflated = zlibCompress(chunk);
            if (deflated.length < chunk.length) {
                writeChunkHeader(out, deflated.length, false);
                out.write(deflated);
            } else {
                writeChunkHeader(out, chunk.length, true);
                out.write(chunk);
            }
            off += n;
        }
        return out.toByteArray();
    }

    public static byte[] decodeCompressed(byte[] file, int offset, int length,
                                          OrcProto.CompressionKind kind,
                                          int blockSize) throws IOException {
        if (length == 0) return new byte[0];
        if (kind == null || kind == OrcProto.CompressionKind.NONE) {
            return Arrays.copyOfRange(file, offset, offset + length);
        }
        if (kind != OrcProto.CompressionKind.ZLIB) {
            throw new IOException("Unsupported ORC compression: " + kind
                + " (pure-Java MVP: NONE/ZLIB only)");
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream(length * 2);
        int end = offset + length;
        int p = offset;
        while (p < end) {
            if (p + 3 > end) throw new IOException("Truncated ORC compression header");
            int b0 = file[p] & 0xff;
            int b1 = file[p + 1] & 0xff;
            int b2 = file[p + 2] & 0xff;
            p += 3;
            int header = b0 | (b1 << 8) | (b2 << 16);
            boolean original = (header & 0x01) != 0;
            int chunkLen = header >>> 1;
            if (p + chunkLen > end) {
                throw new IOException("Truncated ORC compressed chunk");
            }
            if (original) {
                out.write(file, p, chunkLen);
            } else {
                out.write(zlibDecompress(file, p, chunkLen));
            }
            p += chunkLen;
        }
        return out.toByteArray();
    }

    /** Decode a whole compressed buffer (not a slice of a larger file). */
    public static byte[] decodeCompressedBuffer(byte[] buf, OrcProto.CompressionKind kind,
                                                int blockSize) throws IOException {
        if (buf == null || buf.length == 0) return new byte[0];
        return decodeCompressed(buf, 0, buf.length, kind, blockSize);
    }

    private static void writeChunkHeader(ByteArrayOutputStream out, int len, boolean original) {
        int header = (len << 1) | (original ? 1 : 0);
        out.write(header & 0xff);
        out.write((header >>> 8) & 0xff);
        out.write((header >>> 16) & 0xff);
    }

    private static byte[] zlibCompress(byte[] src) {
        Deflater def = new Deflater(Deflater.DEFAULT_COMPRESSION, true); // raw / nowrap like ORC
        try {
            def.setInput(src);
            def.finish();
            byte[] buf = new byte[src.length + 64];
            ByteArrayOutputStream bos = new ByteArrayOutputStream(src.length);
            while (!def.finished()) {
                int n = def.deflate(buf);
                if (n > 0) bos.write(buf, 0, n);
                else break;
            }
            return bos.toByteArray();
        } finally {
            def.end();
        }
    }

    private static byte[] zlibDecompress(byte[] src, int off, int len) throws IOException {
        Inflater inf = new Inflater(true); // nowrap
        try {
            inf.setInput(src, off, len);
            byte[] buf = new byte[Math.max(8192, len * 2)];
            ByteArrayOutputStream bos = new ByteArrayOutputStream(len * 2);
            while (!inf.finished()) {
                int n;
                try {
                    n = inf.inflate(buf);
                } catch (java.util.zip.DataFormatException e) {
                    throw new IOException("ZLIB inflate failed", e);
                }
                if (n > 0) bos.write(buf, 0, n);
                else if (inf.needsInput()) break;
                else break;
            }
            return bos.toByteArray();
        } finally {
            inf.end();
        }
    }

    // ------------------------------------------------------------------ bit / integer RLE

    /** Zigzag encode signed long → unsigned. */
    public static long zigzagEncode(long n) {
        return (n << 1) ^ (n >> 63);
    }

    public static long zigzagDecode(long n) {
        return (n >>> 1) ^ -(n & 1);
    }

    /**
     * Bit-pack PRESENT / boolean bits (MSB first within each byte), then RLE-v1
     * byte-level encoding used by ORC bit streams.
     */
    public static byte[] encodePresentBits(boolean[] present, int count) throws IOException {
        // Pack bits MSB-first into bytes
        int nbytes = (count + 7) / 8;
        byte[] packed = new byte[nbytes];
        for (int i = 0; i < count; i++) {
            if (present[i]) {
                packed[i / 8] |= (byte) (1 << (7 - (i % 8)));
            }
        }
        return encodeByteRle(packed);
    }

    public static boolean[] decodePresentBits(byte[] stream, int count) throws IOException {
        byte[] packed = decodeByteRle(stream, (count + 7) / 8);
        boolean[] present = new boolean[count];
        for (int i = 0; i < count; i++) {
            present[i] = ((packed[i / 8] >> (7 - (i % 8))) & 1) != 0;
        }
        return present;
    }

    /** ORC RunLengthByteWriter / BitFieldWriter underlying byte RLE. */
    public static byte[] encodeByteRle(byte[] data) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(data.length + 8);
        int i = 0;
        while (i < data.length) {
            // Look for runs of identical bytes (min length 3)
            int j = i + 1;
            while (j < data.length && data[j] == data[i] && (j - i) < 130) j++;
            int run = j - i;
            if (run >= 3) {
                // control = run - 3  (0..127 → runs of 3..130)
                out.write(run - 3);
                out.write(data[i] & 0xff);
                i = j;
            } else {
                // literals: collect until a future run of 3+ or 128 literals
                int litStart = i;
                int litCount = 0;
                while (i < data.length && litCount < 128) {
                    // peek upcoming run
                    int k = i + 1;
                    while (k < data.length && data[k] == data[i] && (k - i) < 130) k++;
                    if (k - i >= 3 && litCount > 0) break;
                    if (k - i >= 3 && litCount == 0) break; // will be handled as run
                    i++;
                    litCount++;
                }
                if (litCount == 0) {
                    // pure run starting here
                    continue;
                }
                // control = 256 - litCount  → written as signed byte -litCount
                out.write((-litCount) & 0xff);
                out.write(data, litStart, litCount);
            }
        }
        return out.toByteArray();
    }

    public static byte[] decodeByteRle(byte[] stream, int expectedBytes) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(expectedBytes > 0 ? expectedBytes : 64);
        int p = 0;
        while (p < stream.length && (expectedBytes <= 0 || out.size() < expectedBytes)) {
            int control = stream[p++] & 0xff;
            if (control < 128) {
                // run of (control+3) copies of next byte
                if (p >= stream.length) throw new IOException("Truncated byte RLE run");
                byte val = stream[p++];
                int run = control + 3;
                for (int i = 0; i < run; i++) out.write(val & 0xff);
            } else {
                int lit = 256 - control;
                if (p + lit > stream.length) throw new IOException("Truncated byte RLE literals");
                out.write(stream, p, lit);
                p += lit;
            }
        }
        return out.toByteArray();
    }

    /**
     * Encode signed integers with RLE v2 DIRECT (zigzag + fixed-bit packing).
     * Splits into runs of at most 512 values.
     */
    public static byte[] encodeLongsDirectV2(long[] values, int count) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(count * 4 + 16);
        int i = 0;
        while (i < count) {
            int n = Math.min(512, count - i);
            writeDirectV2Run(out, values, i, n);
            i += n;
        }
        return out.toByteArray();
    }

    private static void writeDirectV2Run(ByteArrayOutputStream out, long[] values,
                                         int off, int n) throws IOException {
        long[] zz = new long[n];
        long max = 0;
        for (int i = 0; i < n; i++) {
            zz[i] = zigzagEncode(values[off + i]);
            if (Long.compareUnsigned(zz[i], max) > 0) max = zz[i];
        }
        int bits = fixedBits(max);
        int encWidth = encodeBitWidth(bits);
        int runLen = n - 1; // 0-based
        // header: 2 bytes
        // bits 7-6: encoding=1 (DIRECT)
        // bits 5-1: encoded bit width
        // bit 0 + next byte: 9-bit run length
        int b0 = (1 << 6) | ((encWidth & 0x1f) << 1) | ((runLen >>> 8) & 0x01);
        int b1 = runLen & 0xff;
        out.write(b0);
        out.write(b1);
        writePackedUnsigned(out, zz, n, bits);
    }

    public static long[] decodeLongsDirectV2(byte[] stream, int count) throws IOException {
        long[] out = new long[count];
        int p = 0;
        int filled = 0;
        while (filled < count && p < stream.length) {
            if (p + 2 > stream.length) throw new IOException("Truncated integer RLE v2 header");
            int b0 = stream[p++] & 0xff;
            int b1 = stream[p++] & 0xff;
            int encoding = (b0 >>> 6) & 0x03;
            if (encoding == 0) {
                // SHORT_REPEAT
                int size = ((b0 >>> 3) & 0x07) + 1; // bytes per value 1..8
                int run = (b0 & 0x07) + 3; // 3..10
                if (p + size > stream.length) throw new IOException("Truncated SHORT_REPEAT");
                long val = readBigEndian(stream, p, size);
                p += size;
                for (int i = 0; i < run && filled < count; i++) {
                    out[filled++] = zigzagDecode(val);
                }
            } else if (encoding == 1) {
                // DIRECT
                int encWidth = (b0 >>> 1) & 0x1f;
                int bits = decodeBitWidth(encWidth);
                int run = ((b0 & 0x01) << 8) | b1;
                run += 1; // 1-based
                int nbytes = (bits * run + 7) / 8;
                if (p + nbytes > stream.length) throw new IOException("Truncated DIRECT run");
                long[] packed = readPackedUnsigned(stream, p, run, bits);
                p += nbytes;
                for (int i = 0; i < run && filled < count; i++) {
                    out[filled++] = zigzagDecode(packed[i]);
                }
            } else if (encoding == 2) {
                // PATCHED_BASE — not fully implemented; best-effort skip not possible without
                // full layout. Fail clearly.
                throw new IOException("ORC integer PATCHED_BASE encoding not supported in MVP");
            } else {
                // DELTA
                int encWidth = (b0 >>> 1) & 0x1f;
                int bits = decodeBitWidth(encWidth);
                int run = ((b0 & 0x01) << 8) | b1;
                run += 1;
                // first value: signed varint (zigzag? no — raw SLONG as bit-packed width of first)
                // Spec: first 2 bytes already consumed; then:
                // - fixed base value width encoded in next bytes as signed BE of width from header's first value width
                // Simplified Apache layout:
                // after header: 1 byte for width of first? Actually:
                //   fb = encoded bit width of deltas
                //   first value is signed vint
                //   then fixed-bit packed deltas
                // We'll implement signed vint + packed deltas.
                LongResult first = readSignedVint(stream, p);
                p = first.pos;
                long prev = first.value;
                if (filled < count) out[filled++] = prev;
                if (run > 1) {
                    // fixed base / delta width
                    int nbytes = (bits * (run - 1) + 7) / 8;
                    if (p + nbytes > stream.length) throw new IOException("Truncated DELTA run");
                    long[] deltas = readPackedUnsigned(stream, p, run - 1, bits);
                    // deltas are zigzag-signed in ORC DELTA? Actually signed via zigzag of delta.
                    p += nbytes;
                    for (int i = 0; i < run - 1 && filled < count; i++) {
                        long d = zigzagDecode(deltas[i]);
                        prev = prev + d;
                        out[filled++] = prev;
                    }
                }
            }
        }
        if (filled < count) {
            throw new IOException("Integer stream produced " + filled + " of " + count + " values");
        }
        return out;
    }

    /** Also support DIRECT (v1) integer encoding used by older writers. */
    public static long[] decodeLongsDirectV1(byte[] stream, int count) throws IOException {
        long[] out = new long[count];
        int p = 0;
        int filled = 0;
        while (filled < count && p < stream.length) {
            int first = stream[p++] & 0xff;
            if (first < 128) {
                // short repeat: 3..130 of next value (signed vint? fixed 1-8?)
                // RLE v1: control < 128 → run of (control+3) of following signed value as vint
                int run = first + 3;
                LongResult r = readSignedVint(stream, p);
                p = r.pos;
                for (int i = 0; i < run && filled < count; i++) out[filled++] = r.value;
            } else {
                int lit = 256 - first; // literals
                for (int i = 0; i < lit && filled < count; i++) {
                    LongResult r = readSignedVint(stream, p);
                    p = r.pos;
                    out[filled++] = r.value;
                }
            }
        }
        if (filled < count) {
            throw new IOException("Integer v1 stream produced " + filled + " of " + count);
        }
        return out;
    }

    public static byte[] encodeLongsDirectV1(long[] values, int count) throws IOException {
        // Use literal runs of up to 128 for simplicity (still valid RLE v1)
        ByteArrayOutputStream out = new ByteArrayOutputStream(count * 2 + 8);
        int i = 0;
        while (i < count) {
            int n = Math.min(128, count - i);
            // Prefer short-repeat when all equal and n>=3
            boolean same = n >= 3;
            if (same) {
                for (int k = 1; k < n; k++) {
                    if (values[i + k] != values[i]) { same = false; break; }
                }
            }
            if (same) {
                out.write(n - 3);
                writeSignedVint(out, values[i]);
                i += n;
            } else {
                out.write((-n) & 0xff);
                for (int k = 0; k < n; k++) writeSignedVint(out, values[i + k]);
                i += n;
            }
        }
        return out.toByteArray();
    }

    // ------------------------------------------------------------------ floats / bytes

    public static byte[] encodeFloats(float[] vals, int count) {
        ByteBuffer bb = ByteBuffer.allocate(count * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) bb.putFloat(vals[i]);
        return bb.array();
    }

    public static float[] decodeFloats(byte[] data, int count) throws IOException {
        if (data.length < count * 4) throw new IOException("Truncated float stream");
        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[count];
        for (int i = 0; i < count; i++) out[i] = bb.getFloat();
        return out;
    }

    public static byte[] encodeDoubles(double[] vals, int count) {
        ByteBuffer bb = ByteBuffer.allocate(count * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) bb.putDouble(vals[i]);
        return bb.array();
    }

    public static double[] decodeDoubles(byte[] data, int count) throws IOException {
        if (data.length < count * 8) throw new IOException("Truncated double stream");
        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        double[] out = new double[count];
        for (int i = 0; i < count; i++) out[i] = bb.getDouble();
        return out;
    }

    public static int formatNanos(long nanos) {
        if (nanos == 0) return 0;
        if (nanos < 0 || nanos > 999_999_999L) {
            nanos = Math.floorMod(nanos, 1_000_000_000L);
        }
        int trailingZeros = 0;
        while (nanos % 100 == 0 && trailingZeros < 7) {
            nanos /= 100;
            trailingZeros++;
        }
        return (int) ((nanos << 3) | trailingZeros);
    }

    public static long parseNanos(long serialized) {
        int zeros = (int) (serialized & 0x07);
        long nanos = serialized >>> 3;
        for (int i = 0; i < zeros; i++) nanos *= 100;
        return nanos;
    }

    // ------------------------------------------------------------------ bit packing helpers

    static int fixedBits(long maxUnsigned) {
        if (maxUnsigned == 0) return 1;
        return 64 - Long.numberOfLeadingZeros(maxUnsigned);
    }

    /** Map actual bit width → 5-bit encoded width (ORC SerializationUtils). */
    static int encodeBitWidth(int n) {
        // ORC only allows certain widths; closest supported:
        // 1..24 map via table; for simplicity use:
        // 0-23 encoded as n-1 for n in 1..24; 26,28,30,32,40,48,56,64 special.
        if (n <= 24) return Math.max(0, n - 1);
        if (n <= 26) return 24;
        if (n <= 28) return 25;
        if (n <= 30) return 26;
        if (n <= 32) return 27;
        if (n <= 40) return 28;
        if (n <= 48) return 29;
        if (n <= 56) return 30;
        return 31; // 64
    }

    static int decodeBitWidth(int encoded) {
        if (encoded <= 23) return encoded + 1;
        switch (encoded) {
            case 24: return 26;
            case 25: return 28;
            case 26: return 30;
            case 27: return 32;
            case 28: return 40;
            case 29: return 48;
            case 30: return 56;
            case 31: return 64;
            default: return encoded + 1;
        }
    }

    static void writePackedUnsigned(ByteArrayOutputStream out, long[] vals, int n, int bits)
        throws IOException {
        if (bits <= 0) bits = 1;
        int totalBits = bits * n;
        int nbytes = (totalBits + 7) / 8;
        byte[] buf = new byte[nbytes];
        int bitPos = 0;
        for (int i = 0; i < n; i++) {
            long v = vals[i];
            for (int b = bits - 1; b >= 0; b--) {
                if (((v >>> b) & 1L) != 0) {
                    buf[bitPos / 8] |= (byte) (1 << (7 - (bitPos % 8)));
                }
                bitPos++;
            }
        }
        out.write(buf);
    }

    static long[] readPackedUnsigned(byte[] data, int offset, int n, int bits) {
        long[] out = new long[n];
        int bitPos = offset * 8;
        // offset is byte offset into data; convert carefully
        int baseBit = 0; // within the slice starting at offset
        for (int i = 0; i < n; i++) {
            long v = 0;
            for (int b = 0; b < bits; b++) {
                int absBit = offset * 8 + baseBit;
                int byteIdx = absBit / 8;
                int bitInByte = 7 - (absBit % 8);
                int bit = (data[byteIdx] >> bitInByte) & 1;
                v = (v << 1) | bit;
                baseBit++;
            }
            out[i] = v;
        }
        return out;
    }

    static long readBigEndian(byte[] data, int off, int size) {
        long v = 0;
        for (int i = 0; i < size; i++) {
            v = (v << 8) | (data[off + i] & 0xffL);
        }
        return v;
    }

    static final class LongResult {
        final long value;
        final int pos;
        LongResult(long value, int pos) { this.value = value; this.pos = pos; }
    }

    /** Signed vint (protobuf-style zigzag LEB128 used by ORC SerializationUtils). */
    static LongResult readSignedVint(byte[] data, int pos) throws IOException {
        LongResult u = readUnsignedVint(data, pos);
        return new LongResult(zigzagDecode(u.value), u.pos);
    }

    static LongResult readUnsignedVint(byte[] data, int pos) throws IOException {
        long result = 0;
        int shift = 0;
        while (pos < data.length) {
            int b = data[pos++] & 0xff;
            result |= (long) (b & 0x7f) << shift;
            if ((b & 0x80) == 0) return new LongResult(result, pos);
            shift += 7;
            if (shift > 63) throw new IOException("vint too long");
        }
        throw new IOException("Truncated vint");
    }

    static void writeSignedVint(ByteArrayOutputStream out, long v) {
        writeUnsignedVint(out, zigzagEncode(v));
    }

    static void writeUnsignedVint(ByteArrayOutputStream out, long v) {
        while ((v & ~0x7fL) != 0) {
            out.write((int) ((v & 0x7f) | 0x80));
            v >>>= 7;
        }
        out.write((int) v);
    }

    // ------------------------------------------------------------------ misc

    public static String utf8(byte[] b) {
        return new String(b, StandardCharsets.UTF_8);
    }

    public static byte[] utf8(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    public static List<OrcProto.Type> emptyTypes() {
        return new ArrayList<>();
    }
}
