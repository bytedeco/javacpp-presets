package org.bytedeco.pytorch.data.orc;

import org.apache.orc.OrcProto;
import org.bytedeco.pytorch.dataframe.Column;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure-Java ORC file reader (no Hadoop / no {@code orc-core}).
 *
 * <p>Reads ORC v0.12-compatible files written by {@link OrcOutputFormat} and
 * (for MVP primitives) common DIRECT / DIRECT_V2 streams from other writers.
 *
 * <pre>
 *   try (OrcInputFormat in = OrcInputFormat.open(path)) {
 *       Object[] row;
 *       while ((row = in.read()) != null) { ... }
 *   }
 * </pre>
 */
public final class OrcInputFormat implements Closeable {

    private final Path path;
    private final byte[] file;
    private final OrcFormatCodec.FileTail tail;
    private final OrcTypeMapper.Schema schema;
    private final OrcProto.CompressionKind compression;
    private final int compressionBlockSize;

    private int stripeIndex;
    private long remainingInStripe;
    private int rowInStripe;
    private Object[][] stripeRows; // [row][col]
    private boolean closed;

    private OrcInputFormat(Path path, byte[] file, OrcFormatCodec.FileTail tail,
                           OrcTypeMapper.Schema schema) {
        this.path = path;
        this.file = file;
        this.tail = tail;
        this.schema = schema;
        this.compression = tail.postScript.hasCompression()
            ? tail.postScript.getCompression() : OrcProto.CompressionKind.NONE;
        this.compressionBlockSize = tail.postScript.hasCompressionBlockSize()
            ? (int) tail.postScript.getCompressionBlockSize()
            : OrcFormatCodec.DEFAULT_COMPRESSION_BLOCK;
        this.stripeIndex = 0;
        this.remainingInStripe = 0;
        this.rowInStripe = 0;
    }

    public static OrcInputFormat open(String path) throws IOException {
        return open(Paths.get(path));
    }

    public static OrcInputFormat open(Path path) throws IOException {
        byte[] all = Files.readAllBytes(path);
        OrcFormatCodec.FileTail tail = OrcFormatCodec.readFileTail(all);
        OrcTypeMapper.Schema schema = OrcTypeMapper.fromFooter(tail.footer);
        return new OrcInputFormat(path, all, tail, schema);
    }

    public OrcTypeMapper.Schema schema() { return schema; }

    /** Top-level field names in schema order (aligned with {@link #read()}). */
    public java.util.List<String> fieldNames() {
        java.util.List<String> names = new java.util.ArrayList<>(schema.fields.size());
        for (OrcTypeMapper.Field f : schema.fields) names.add(f.name);
        return names;
    }

    public OrcProto.Footer footer() { return tail.footer; }

    public long numberOfRows() {
        return tail.footer.hasNumberOfRows() ? tail.footer.getNumberOfRows() : -1L;
    }

    /**
     * Read next row as Object[] aligned to {@link #schema()} fields, or {@code null} at EOF.
     */
    public Object[] read() throws IOException {
        ensureOpen();
        if (remainingInStripe <= 0) {
            if (!loadNextStripe()) return null;
        }
        Object[] row = stripeRows[rowInStripe];
        rowInStripe++;
        remainingInStripe--;
        return row;
    }

    private boolean loadNextStripe() throws IOException {
        List<OrcProto.StripeInformation> stripes = tail.footer.getStripesList();
        while (stripeIndex < stripes.size()) {
            OrcProto.StripeInformation si = stripes.get(stripeIndex++);
            long rows = si.getNumberOfRows();
            if (rows <= 0) continue;
            stripeRows = decodeStripe(si);
            rowInStripe = 0;
            remainingInStripe = rows;
            return true;
        }
        return false;
    }

    private Object[][] decodeStripe(OrcProto.StripeInformation si) throws IOException {
        int nRows = (int) si.getNumberOfRows();
        long offset = si.getOffset();
        long indexLen = si.hasIndexLength() ? si.getIndexLength() : 0L;
        long dataLen = si.getDataLength();
        long footerLen = si.getFooterLength();

        // Stripe footer sits after index+data
        int footerStart = (int) (offset + indexLen + dataLen);
        byte[] footerBytes = OrcFormatCodec.decodeCompressed(
            file, footerStart, (int) footerLen, compression, compressionBlockSize);
        OrcProto.StripeFooter stripeFooter = OrcProto.StripeFooter.parseFrom(footerBytes);

        List<OrcProto.ColumnEncoding> encodings = stripeFooter.getColumnsList();
        List<OrcProto.Stream> streams = stripeFooter.getStreamsList();

        // Map columnId → list of streams in order
        Map<Integer, List<StreamSlice>> byCol = new HashMap<>();
        long cursor = offset + indexLen;
        for (OrcProto.Stream s : streams) {
            int col = s.getColumn();
            long len = s.getLength();
            StreamSlice slice = new StreamSlice(s.getKind(), (int) cursor, (int) len);
            byCol.computeIfAbsent(col, k -> new ArrayList<>()).add(slice);
            cursor += len;
        }

        int nFields = schema.fields.size();
        Object[][] cols = new Object[nFields][];

        for (int f = 0; f < nFields; f++) {
            OrcTypeMapper.Field field = schema.fields.get(f);
            int colId = field.columnId;
            OrcProto.ColumnEncoding.Kind enc = encodingOf(encodings, colId);
            List<StreamSlice> slices = byCol.getOrDefault(colId, List.of());
            if (field.isList()) {
                OrcProto.ColumnEncoding.Kind elemEnc = encodingOf(encodings, field.elementColumnId);
                List<StreamSlice> elemSlices = byCol.getOrDefault(field.elementColumnId, List.of());
                cols[f] = decodeListColumn(field, enc, slices, elemEnc, elemSlices, nRows);
            } else {
                cols[f] = decodeLeafColumn(field.kind, enc, slices, nRows, field.dtype);
            }
        }

        // Transpose to row-major
        Object[][] rows = new Object[nRows][nFields];
        for (int r = 0; r < nRows; r++) {
            for (int c = 0; c < nFields; c++) {
                rows[r][c] = cols[c][r];
            }
        }
        return rows;
    }

    private static OrcProto.ColumnEncoding.Kind encodingOf(
            List<OrcProto.ColumnEncoding> encodings, int colId) {
        if (colId >= 0 && colId < encodings.size() && encodings.get(colId).hasKind()) {
            return encodings.get(colId).getKind();
        }
        return OrcProto.ColumnEncoding.Kind.DIRECT;
    }

    private static final class StreamSlice {
        final OrcProto.Stream.Kind kind;
        final int offset;
        final int length;
        StreamSlice(OrcProto.Stream.Kind kind, int offset, int length) {
            this.kind = kind;
            this.offset = offset;
            this.length = length;
        }
    }

    private byte[] loadStream(StreamSlice s) throws IOException {
        if (s == null || s.length == 0) return new byte[0];
        return OrcFormatCodec.decodeCompressed(
            file, s.offset, s.length, compression, compressionBlockSize);
    }

    private StreamSlice find(List<StreamSlice> slices, OrcProto.Stream.Kind kind) {
        for (StreamSlice s : slices) {
            if (s.kind == kind) return s;
        }
        return null;
    }

    private boolean[] readPresent(List<StreamSlice> slices, int count) throws IOException {
        StreamSlice presentSlice = find(slices, OrcProto.Stream.Kind.PRESENT);
        if (presentSlice == null || presentSlice.length == 0) {
            boolean[] present = new boolean[count];
            Arrays.fill(present, true);
            return present;
        }
        return OrcFormatCodec.decodePresentBits(loadStream(presentSlice), count);
    }

    private Object[] decodeListColumn(OrcTypeMapper.Field field,
                                      OrcProto.ColumnEncoding.Kind listEnc,
                                      List<StreamSlice> listSlices,
                                      OrcProto.ColumnEncoding.Kind elemEnc,
                                      List<StreamSlice> elemSlices,
                                      int nRows) throws IOException {
        boolean[] present = readPresent(listSlices, nRows);
        int nonNull = 0;
        for (boolean p : present) if (p) nonNull++;

        Object[] out = new Object[nRows];
        if (nonNull == 0) return out;

        StreamSlice lengthSlice = find(listSlices, OrcProto.Stream.Kind.LENGTH);
        long[] lens = decodeIntegerStream(loadStream(lengthSlice), listEnc, nonNull);
        long totalElemsLong = 0;
        for (long l : lens) totalElemsLong += l;
        if (totalElemsLong > Integer.MAX_VALUE) {
            throw new IOException("LIST element count overflow: " + totalElemsLong);
        }
        int totalElems = (int) totalElemsLong;

        // Decode element column for totalElems rows
        Object[] flat = decodeLeafColumn(field.elementKind, elemEnc, elemSlices,
            totalElems, field.elementDtype);

        int elemCursor = 0;
        int lenCursor = 0;
        for (int r = 0; r < nRows; r++) {
            if (!present[r]) continue;
            int len = (int) lens[lenCursor++];
            if (elemCursor + len > flat.length) {
                throw new IOException("LIST length overrun at row " + r);
            }
            List<Object> elems = new ArrayList<>(len);
            for (int i = 0; i < len; i++) elems.add(flat[elemCursor++]);
            out[r] = OrcTypeMapper.densify(elems, field.elementDtype, field.dtype);
        }
        return out;
    }

    private Object[] decodeLeafColumn(OrcProto.Type.Kind kind,
                                      OrcProto.ColumnEncoding.Kind enc,
                                      List<StreamSlice> slices,
                                      int nRows,
                                      Column.DType preferDtype)
            throws IOException {
        boolean[] present = readPresent(slices, nRows);
        int nonNull = 0;
        for (boolean p : present) if (p) nonNull++;

        Object[] out = new Object[nRows];
        if (nonNull == 0) return out;

        StreamSlice dataSlice = find(slices, OrcProto.Stream.Kind.DATA);
        StreamSlice lengthSlice = find(slices, OrcProto.Stream.Kind.LENGTH);
        StreamSlice secondarySlice = find(slices, OrcProto.Stream.Kind.SECONDARY);

        switch (kind) {
            case BOOLEAN: {
                byte[] data = loadStream(dataSlice);
                boolean[] bits = OrcFormatCodec.decodePresentBits(data, nonNull);
                int j = 0;
                for (int r = 0; r < nRows; r++) {
                    if (!present[r]) continue;
                    out[r] = bits[j++];
                }
                return out;
            }
            case BYTE:
            case SHORT:
            case INT:
            case LONG:
            case DATE: {
                long[] vals = decodeIntegerStream(loadStream(dataSlice), enc, nonNull);
                int j = 0;
                for (int r = 0; r < nRows; r++) {
                    if (!present[r]) continue;
                    long v = vals[j++];
                    if (kind == OrcProto.Type.Kind.DATE) {
                        out[r] = LocalDate.ofEpochDay(v);
                    } else if (kind == OrcProto.Type.Kind.LONG
                        || preferDtype == Column.DType.INT64) {
                        out[r] = v;
                    } else {
                        out[r] = (int) v;
                    }
                }
                return out;
            }
            case FLOAT: {
                float[] vals = OrcFormatCodec.decodeFloats(loadStream(dataSlice), nonNull);
                int j = 0;
                for (int r = 0; r < nRows; r++) {
                    if (!present[r]) continue;
                    out[r] = vals[j++];
                }
                return out;
            }
            case DOUBLE: {
                double[] vals = OrcFormatCodec.decodeDoubles(loadStream(dataSlice), nonNull);
                int j = 0;
                for (int r = 0; r < nRows; r++) {
                    if (!present[r]) continue;
                    out[r] = vals[j++];
                }
                return out;
            }
            case STRING:
            case VARCHAR:
            case CHAR:
            case BINARY: {
                byte[] data = loadStream(dataSlice);
                long[] lens = decodeIntegerStream(loadStream(lengthSlice), enc, nonNull);
                int pos = 0;
                int j = 0;
                for (int r = 0; r < nRows; r++) {
                    if (!present[r]) continue;
                    int len = (int) lens[j++];
                    if (pos + len > data.length) {
                        throw new IOException("String/binary data overrun");
                    }
                    byte[] slice = Arrays.copyOfRange(data, pos, pos + len);
                    pos += len;
                    if (kind == OrcProto.Type.Kind.BINARY) {
                        out[r] = slice;
                    } else {
                        out[r] = OrcFormatCodec.utf8(slice);
                    }
                }
                return out;
            }
            case TIMESTAMP:
            case TIMESTAMP_INSTANT: {
                long[] seconds = decodeIntegerStream(loadStream(dataSlice), enc, nonNull);
                long[] nanosRaw = decodeIntegerStream(loadStream(secondarySlice), enc, nonNull);
                final long ORC_EPOCH = Instant.parse("2015-01-01T00:00:00Z").getEpochSecond();
                int j = 0;
                for (int r = 0; r < nRows; r++) {
                    if (!present[r]) continue;
                    long sec = seconds[j] + ORC_EPOCH;
                    long nanos = OrcFormatCodec.parseNanos(nanosRaw[j]);
                    j++;
                    out[r] = Instant.ofEpochSecond(sec, nanos);
                }
                return out;
            }
            default:
                throw new UnsupportedOperationException(
                    "ORC read unsupported kind: " + kind);
        }
    }

    private static long[] decodeIntegerStream(byte[] stream,
                                              OrcProto.ColumnEncoding.Kind enc,
                                              int count) throws IOException {
        if (count == 0) return new long[0];
        if (stream == null || stream.length == 0) {
            throw new IOException("Missing integer data stream for " + count + " values");
        }
        if (enc == OrcProto.ColumnEncoding.Kind.DIRECT_V2
            || enc == OrcProto.ColumnEncoding.Kind.DICTIONARY_V2) {
            return OrcFormatCodec.decodeLongsDirectV2(stream, count);
        }
        // DIRECT / DICTIONARY (v1)
        return OrcFormatCodec.decodeLongsDirectV1(stream, count);
    }

    private void ensureOpen() throws IOException {
        if (closed) throw new IOException("OrcInputFormat is closed");
    }

    @Override
    public void close() {
        closed = true;
        stripeRows = null;
    }
}
