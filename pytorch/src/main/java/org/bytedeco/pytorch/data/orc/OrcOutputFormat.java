package org.bytedeco.pytorch.data.orc;

import org.apache.orc.OrcProto;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec;

/**
 * Pure-Java ORC file writer (no Hadoop / no {@code orc-core}).
 *
 * <p>Writes ORC v0.12-compatible files using {@link OrcProto} from {@code orc-format}.
 * MVP: flat STRUCT of primitives, DIRECT / DIRECT_V2 encodings, compression NONE or ZLIB.
 *
 * <pre>
 *   try (OrcOutputFormat out = OrcOutputFormat.builder(path, schema)
 *           .withCompression(OrcProto.CompressionKind.NONE)
 *           .build()) {
 *       out.writeRow(new Object[]{1, "alice", true});
 *   }
 * </pre>
 */
public final class OrcOutputFormat implements Closeable {

    public static final class Builder {
        private final String path;
        private final OrcTypeMapper.Schema schema;
        private OrcProto.CompressionKind compression = OrcProto.CompressionKind.NONE;
        private int compressionBlockSize = OrcFormatCodec.DEFAULT_COMPRESSION_BLOCK;
        private long stripeRows = 10_000; // soft row target per stripe
        private long stripeBytes = 64L * 1024 * 1024;
        private boolean overwrite = true;

        public Builder(String path, OrcTypeMapper.Schema schema) {
            this.path = path;
            this.schema = schema;
        }

        public Builder withCompression(OrcProto.CompressionKind kind) {
            this.compression = kind == null ? OrcProto.CompressionKind.NONE : kind;
            return this;
        }

        public Builder withCompressionBlockSize(int bytes) {
            this.compressionBlockSize = bytes > 0 ? bytes : OrcFormatCodec.DEFAULT_COMPRESSION_BLOCK;
            return this;
        }

        public Builder withStripeRows(long rows) {
            this.stripeRows = rows > 0 ? rows : 10_000;
            return this;
        }

        public Builder withStripeBytes(long bytes) {
            this.stripeBytes = bytes > 0 ? bytes : 64L * 1024 * 1024;
            return this;
        }

        public Builder withOverwrite(boolean overwrite) {
            this.overwrite = overwrite;
            return this;
        }

        public OrcOutputFormat build() throws IOException {
            return new OrcOutputFormat(this);
        }
    }

    public static Builder builder(String path, OrcTypeMapper.Schema schema) {
        return new Builder(path, schema);
    }

    private final Path path;
    private final OrcTypeMapper.Schema schema;
    private final OrcProto.CompressionKind compression;
    private final int compressionBlockSize;
    private final long stripeRows;
    private final long stripeBytes;

    private OutputStream out;
    private long fileOffset;
    private long totalRows;
    private boolean closed;

    // current stripe buffers (column-major Object lists)
    private final List<List<Object>> stripeCols;
    private int stripeRowCount;
    private long stripeEstimateBytes;

    private final List<OrcProto.StripeInformation> stripes = new ArrayList<>();

    private OrcOutputFormat(Builder b) throws IOException {
        this.path = Paths.get(b.path);
        this.schema = b.schema;
        this.compression = b.compression;
        this.compressionBlockSize = b.compressionBlockSize;
        this.stripeRows = b.stripeRows;
        this.stripeBytes = b.stripeBytes;

        if (Files.exists(path)) {
            if (!b.overwrite) {
                throw new IllegalStateException("ORC file exists and overwrite=false: " + path);
            }
            Files.delete(path);
        } else {
            Path parent = path.getParent();
            if (parent != null) Files.createDirectories(parent);
        }

        this.out = Files.newOutputStream(path);
        out.write(OrcFormatCodec.MAGIC);
        this.fileOffset = OrcFormatCodec.MAGIC.length;

        int n = schema.fields.size();
        this.stripeCols = new ArrayList<>(n);
        for (int i = 0; i < n; i++) stripeCols.add(new ArrayList<>());
        this.stripeRowCount = 0;
        this.stripeEstimateBytes = 0;
    }

    public OrcTypeMapper.Schema schema() { return schema; }

    public void writeRow(Object[] row) throws IOException {
        ensureOpen();
        if (row == null) throw new IllegalArgumentException("row required");
        if (row.length != schema.fields.size()) {
            throw new IllegalArgumentException(
                "row width " + row.length + " != schema " + schema.fields.size());
        }
        for (int i = 0; i < row.length; i++) {
            stripeCols.get(i).add(row[i]);
            stripeEstimateBytes += estimateSize(row[i]);
        }
        stripeRowCount++;
        totalRows++;
        if (stripeRowCount >= stripeRows || stripeEstimateBytes >= stripeBytes) {
            flushStripe();
        }
    }

    private static int estimateSize(Object v) {
        if (v == null) return 1;
        if (v instanceof String) return ((String) v).length() + 4;
        if (v instanceof byte[]) return ((byte[]) v).length + 4;
        if (v instanceof long[]) return ((long[]) v).length * 8 + 8;
        if (v instanceof int[]) return ((int[]) v).length * 4 + 8;
        if (v instanceof float[]) return ((float[]) v).length * 4 + 8;
        if (v instanceof double[]) return ((double[]) v).length * 8 + 8;
        if (v instanceof List) return ((List<?>) v).size() * 8 + 8;
        if (v instanceof Number || v instanceof Boolean) return 8;
        return 16;
    }

    private void flushStripe() throws IOException {
        if (stripeRowCount == 0) return;
        long stripeOffset = fileOffset;
        long indexLength = 0; // no row indexes in MVP
        long dataLength = 0;

        List<OrcProto.Stream> streams = new ArrayList<>();
        // encodings indexed by column id (dense 0..maxId)
        int nTypes = schema.types.size();
        OrcProto.ColumnEncoding[] encById = new OrcProto.ColumnEncoding[nTypes];
        encById[0] = OrcProto.ColumnEncoding.newBuilder()
            .setKind(OrcProto.ColumnEncoding.Kind.DIRECT).build();

        int nFields = schema.fields.size();
        for (int f = 0; f < nFields; f++) {
            OrcTypeMapper.Field field = schema.fields.get(f);
            List<Object> col = stripeCols.get(f);
            int colId = field.columnId;

            boolean[] present = new boolean[stripeRowCount];
            int nonNull = 0;
            for (int r = 0; r < stripeRowCount; r++) {
                present[r] = col.get(r) != null;
                if (present[r]) nonNull++;
            }
            dataLength += writeStream(streams, OrcProto.Stream.Kind.PRESENT, colId,
                OrcFormatCodec.encodePresentBits(present, stripeRowCount));

            if (field.isList()) {
                // LIST column: LENGTH only (no DATA). DIRECT_V2 for integer lengths.
                encById[colId] = OrcProto.ColumnEncoding.newBuilder()
                    .setKind(OrcProto.ColumnEncoding.Kind.DIRECT_V2).build();

                long[] lens = new long[nonNull];
                List<Object> flatElems = new ArrayList<>();
                int j = 0;
                for (int r = 0; r < stripeRowCount; r++) {
                    if (!present[r]) continue;
                    Object[] elems = OrcTypeMapper.flattenListCell(col.get(r));
                    if (elems == null) elems = new Object[0];
                    lens[j++] = elems.length;
                    for (Object e : elems) flatElems.add(e);
                }
                boolean[] elemPresent = new boolean[flatElems.size()];
                for (int i = 0; i < flatElems.size(); i++) {
                    elemPresent[i] = flatElems.get(i) != null;
                }
                dataLength += writeStream(streams, OrcProto.Stream.Kind.LENGTH, colId,
                    OrcFormatCodec.encodeLongsDirectV2(lens, nonNull));

                int elemId = field.elementColumnId;
                int elemNonNull = 0;
                for (boolean p : elemPresent) if (p) elemNonNull++;
                dataLength += writeStream(streams, OrcProto.Stream.Kind.PRESENT, elemId,
                    OrcFormatCodec.encodePresentBits(elemPresent,
                        Math.max(flatElems.size(), 0)));

                EncodedColumn elemEnc = encodeScalarValues(
                    field.elementKind, flatElems, elemPresent, elemNonNull);
                encById[elemId] = OrcProto.ColumnEncoding.newBuilder()
                    .setKind(elemEnc.encoding).build();
                dataLength += writeEncodedStreams(streams, elemId, elemEnc);
            } else {
                EncodedColumn encoded = encodeScalarValues(field.kind, col, present, nonNull);
                encById[colId] = OrcProto.ColumnEncoding.newBuilder()
                    .setKind(encoded.encoding).build();
                dataLength += writeEncodedStreams(streams, colId, encoded);
            }
        }

        // Fill any missing encoding slots (should not happen) with DIRECT
        List<OrcProto.ColumnEncoding> encodings = new ArrayList<>(nTypes);
        for (int i = 0; i < nTypes; i++) {
            encodings.add(encById[i] != null ? encById[i]
                : OrcProto.ColumnEncoding.newBuilder()
                    .setKind(OrcProto.ColumnEncoding.Kind.DIRECT).build());
        }

        OrcProto.StripeFooter footer = OrcProto.StripeFooter.newBuilder()
            .addAllStreams(streams)
            .addAllColumns(encodings)
            .setWriterTimezone("UTC")
            .build();
        byte[] footerRaw = footer.toByteArray();
        byte[] footerBytes = OrcFormatCodec.encodeCompressed(
            footerRaw, compression, compressionBlockSize);
        out.write(footerBytes);
        long footerLength = footerBytes.length;
        fileOffset += footerLength;

        stripes.add(OrcProto.StripeInformation.newBuilder()
            .setOffset(stripeOffset)
            .setIndexLength(indexLength)
            .setDataLength(dataLength)
            .setFooterLength(footerLength)
            .setNumberOfRows(stripeRowCount)
            .build());

        for (List<Object> c : stripeCols) c.clear();
        stripeRowCount = 0;
        stripeEstimateBytes = 0;
    }

    private long writeStream(List<OrcProto.Stream> streams, OrcProto.Stream.Kind kind,
                             int columnId, byte[] raw) throws IOException {
        if (raw == null) raw = new byte[0];
        byte[] bytes = OrcFormatCodec.encodeCompressed(raw, compression, compressionBlockSize);
        out.write(bytes);
        fileOffset += bytes.length;
        streams.add(OrcProto.Stream.newBuilder()
            .setKind(kind)
            .setColumn(columnId)
            .setLength(bytes.length)
            .build());
        return bytes.length;
    }

    private long writeEncodedStreams(List<OrcProto.Stream> streams, int colId,
                                     EncodedColumn encoded) throws IOException {
        long n = 0;
        if (encoded.data != null && encoded.data.length > 0) {
            n += writeStream(streams, OrcProto.Stream.Kind.DATA, colId, encoded.data);
        }
        if (encoded.length != null && encoded.length.length > 0) {
            n += writeStream(streams, OrcProto.Stream.Kind.LENGTH, colId, encoded.length);
        }
        if (encoded.secondary != null && encoded.secondary.length > 0) {
            n += writeStream(streams, OrcProto.Stream.Kind.SECONDARY, colId, encoded.secondary);
        }
        return n;
    }

    private static final class EncodedColumn {
        final OrcProto.ColumnEncoding.Kind encoding;
        final byte[] data;
        final byte[] length;
        final byte[] secondary;

        EncodedColumn(OrcProto.ColumnEncoding.Kind encoding, byte[] data,
                      byte[] length, byte[] secondary) {
            this.encoding = encoding;
            this.data = data;
            this.length = length;
            this.secondary = secondary;
        }
    }

    /**
     * Encode scalar (or flattened list-element) values for non-null positions.
     * {@code values} may be a column list or a flat element list.
     */
    private EncodedColumn encodeScalarValues(OrcProto.Type.Kind kind, List<Object> values,
                                             boolean[] present, int nonNull) throws IOException {
        switch (kind) {
            case BOOLEAN: {
                boolean[] bits = new boolean[nonNull];
                int j = 0;
                for (int r = 0; r < values.size(); r++) {
                    if (!present[r]) continue;
                    bits[j++] = toBoolean(values.get(r));
                }
                byte[] data = OrcFormatCodec.encodePresentBits(bits, nonNull);
                return new EncodedColumn(OrcProto.ColumnEncoding.Kind.DIRECT, data, null, null);
            }
            case BYTE:
            case SHORT:
            case INT:
            case LONG:
            case DATE: {
                long[] vals = new long[nonNull];
                int j = 0;
                for (int r = 0; r < values.size(); r++) {
                    if (!present[r]) continue;
                    vals[j++] = toLong(values.get(r), kind);
                }
                byte[] data = OrcFormatCodec.encodeLongsDirectV2(vals, nonNull);
                return new EncodedColumn(OrcProto.ColumnEncoding.Kind.DIRECT_V2, data, null, null);
            }
            case FLOAT: {
                float[] vals = new float[nonNull];
                int j = 0;
                for (int r = 0; r < values.size(); r++) {
                    if (!present[r]) continue;
                    vals[j++] = toFloat(values.get(r));
                }
                return new EncodedColumn(OrcProto.ColumnEncoding.Kind.DIRECT,
                    OrcFormatCodec.encodeFloats(vals, nonNull), null, null);
            }
            case DOUBLE: {
                double[] vals = new double[nonNull];
                int j = 0;
                for (int r = 0; r < values.size(); r++) {
                    if (!present[r]) continue;
                    vals[j++] = toDouble(values.get(r));
                }
                return new EncodedColumn(OrcProto.ColumnEncoding.Kind.DIRECT,
                    OrcFormatCodec.encodeDoubles(vals, nonNull), null, null);
            }
            case STRING:
            case VARCHAR:
            case CHAR:
            case BINARY: {
                long[] lens = new long[nonNull];
                ByteArrayOutputStream data = new ByteArrayOutputStream();
                int j = 0;
                for (int r = 0; r < values.size(); r++) {
                    if (!present[r]) continue;
                    byte[] bytes = toBytes(values.get(r), kind);
                    lens[j++] = bytes.length;
                    data.write(bytes);
                }
                byte[] lengthStream = OrcFormatCodec.encodeLongsDirectV2(lens, nonNull);
                return new EncodedColumn(OrcProto.ColumnEncoding.Kind.DIRECT_V2,
                    data.toByteArray(), lengthStream, null);
            }
            case TIMESTAMP:
            case TIMESTAMP_INSTANT: {
                long[] seconds = new long[nonNull];
                long[] nanos = new long[nonNull];
                int j = 0;
                final long ORC_EPOCH = Instant.parse("2015-01-01T00:00:00Z").getEpochSecond();
                for (int r = 0; r < values.size(); r++) {
                    if (!present[r]) continue;
                    Instant inst = toInstant(values.get(r));
                    seconds[j] = inst.getEpochSecond() - ORC_EPOCH;
                    nanos[j] = OrcFormatCodec.formatNanos(inst.getNano());
                    j++;
                }
                return new EncodedColumn(OrcProto.ColumnEncoding.Kind.DIRECT_V2,
                    OrcFormatCodec.encodeLongsDirectV2(seconds, nonNull),
                    null,
                    OrcFormatCodec.encodeLongsDirectV2(nanos, nonNull));
            }
            default:
                throw new UnsupportedOperationException(
                    "ORC write unsupported kind: " + kind);
        }
    }

    private static boolean toBoolean(Object v) {
        if (v instanceof Boolean) return (Boolean) v;
        return Boolean.parseBoolean(String.valueOf(v));
    }

    private static long toLong(Object v, OrcProto.Type.Kind kind) {
        if (kind == OrcProto.Type.Kind.DATE) {
            if (v instanceof LocalDate) return ((LocalDate) v).toEpochDay();
            if (v instanceof Number) return ((Number) v).longValue();
            return LocalDate.parse(String.valueOf(v)).toEpochDay();
        }
        if (v instanceof Number) return ((Number) v).longValue();
        return Long.parseLong(String.valueOf(v));
    }

    private static float toFloat(Object v) {
        if (v instanceof Number) return ((Number) v).floatValue();
        return Float.parseFloat(String.valueOf(v));
    }

    private static double toDouble(Object v) {
        if (v instanceof Number) return ((Number) v).doubleValue();
        return Double.parseDouble(String.valueOf(v));
    }

    private static byte[] toBytes(Object v, OrcProto.Type.Kind kind) {
        if (v instanceof byte[]) return (byte[]) v;
        if (kind == OrcProto.Type.Kind.BINARY) {
            if (v instanceof Map || v instanceof List || v.getClass().isArray()) {
                return ComplexCellCodec.encodeText(v).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            }
            return String.valueOf(v).getBytes(java.nio.charset.StandardCharsets.UTF_8);
        }
        // STRING path: nested cells → JSON text
        if (v instanceof Map || v instanceof List || v.getClass().isArray()) {
            return OrcFormatCodec.utf8(ComplexCellCodec.encodeText(v));
        }
        return OrcFormatCodec.utf8(String.valueOf(v));
    }

    private static Instant toInstant(Object v) {
        if (v instanceof Instant) return (Instant) v;
        if (v instanceof LocalDateTime) {
            return ((LocalDateTime) v).toInstant(ZoneOffset.UTC);
        }
        if (v instanceof Number) {
            // treat as epoch millis
            return Instant.ofEpochMilli(((Number) v).longValue());
        }
        return Instant.parse(String.valueOf(v));
    }

    private void writeTail() throws IOException {
        if (stripeRowCount > 0) flushStripe();

        // Metadata (empty stripe stats OK)
        OrcProto.Metadata metadata = OrcProto.Metadata.newBuilder().build();
        byte[] metaRaw = metadata.toByteArray();
        byte[] metaBytes = OrcFormatCodec.encodeCompressed(
            metaRaw, compression, compressionBlockSize);

        // Footer
        long headerLength = OrcFormatCodec.MAGIC.length;
        long contentLength = fileOffset; // after last stripe footer
        OrcProto.Footer.Builder fb = OrcProto.Footer.newBuilder()
            .setHeaderLength(headerLength)
            .setContentLength(contentLength)
            .setNumberOfRows(totalRows)
            .setRowIndexStride(0)
            .setWriter(OrcFormatCodec.WRITER_ID)
            .setSoftwareVersion(OrcFormatCodec.SOFTWARE_VERSION)
            .addAllTypes(schema.types)
            .addAllStripes(stripes);
        byte[] footerRaw = fb.build().toByteArray();
        byte[] footerBytes = OrcFormatCodec.encodeCompressed(
            footerRaw, compression, compressionBlockSize);

        out.write(metaBytes);
        out.write(footerBytes);
        fileOffset += metaBytes.length + footerBytes.length;

        byte[] ps = OrcFormatCodec.buildPostScript(
            footerBytes.length, metaBytes.length, compression, compressionBlockSize);
        if (ps.length > 255) {
            throw new IOException("PostScript too large: " + ps.length);
        }
        out.write(ps);
        out.write(ps.length & 0xff);
        fileOffset += ps.length + 1;
    }

    private void ensureOpen() throws IOException {
        if (closed) throw new IOException("OrcOutputFormat is closed");
    }

    @Override
    public void close() throws IOException {
        if (closed) return;
        try {
            writeTail();
            out.flush();
        } finally {
            closed = true;
            if (out != null) {
                out.close();
                out = null;
            }
        }
    }
}
