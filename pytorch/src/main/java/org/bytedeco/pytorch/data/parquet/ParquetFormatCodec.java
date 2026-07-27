package org.bytedeco.pytorch.data.parquet;

import org.apache.parquet.column.Encoding;
import org.apache.parquet.column.page.DataPageV1;
import org.apache.parquet.column.page.DictionaryPage;
import org.apache.parquet.format.CompressionCodec;
import org.apache.parquet.format.ConvertedType;
import org.apache.parquet.format.DataPageHeader;
import org.apache.parquet.format.DictionaryPageHeader;
import org.apache.parquet.format.FieldRepetitionType;
import org.apache.parquet.format.FileMetaData;
import org.apache.parquet.format.LogicalType;
import org.apache.parquet.format.PageHeader;
import org.apache.parquet.format.PageType;
import org.apache.parquet.format.SchemaElement;
import org.apache.parquet.format.StringType;
import org.apache.parquet.format.Type;
import org.apache.parquet.format.Util;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.schema.GroupType;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.OriginalType;
import org.apache.parquet.schema.PrimitiveType;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;
import org.apache.parquet.schema.Type.Repetition;
import org.apache.parquet.schema.Types;
import org.xerial.snappy.Snappy;

import com.github.luben.zstd.Zstd;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Pure-Java helpers for Parquet thrift footer / page headers / compression /
 * schema conversion. No Hadoop, no {@code parquet-hadoop}.
 */
public final class ParquetFormatCodec {
    public static final byte[] MAGIC = new byte[]{'P', 'A', 'R', '1'};
    public static final String CREATED_BY = "org.bytedeco.pytorch.data.parquet";

    private ParquetFormatCodec() {}

    // ---- magic / footer framing --------------------------------------------

    public static void writeMagic(OutputStream out) throws IOException {
        out.write(MAGIC);
    }

    public static void verifyMagic(byte[] b, int off) throws IOException {
        if (b.length < off + 4
            || b[off] != 'P' || b[off + 1] != 'A' || b[off + 2] != 'R' || b[off + 3] != '1') {
            throw new IOException("Not a Parquet file (bad magic)");
        }
    }

    /** Read footer from local path: returns thrift FileMetaData. */
    public static FileMetaData readFileMetaData(Path path) throws IOException {
        long len = Files.size(path);
        if (len < 8) throw new IOException("File too small for Parquet: " + path);
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            // tail: 4-byte LE footer length + MAGIC
            ByteBuffer tail = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            ch.read(tail, len - 8);
            tail.flip();
            int footerLen = tail.getInt();
            byte[] magic = new byte[4];
            tail.get(magic);
            verifyMagic(magic, 0);
            if (footerLen <= 0 || footerLen > len - 8) {
                throw new IOException("Invalid Parquet footer length: " + footerLen);
            }
            long footerStart = len - 8 - footerLen;
            // also verify head magic
            ByteBuffer head = ByteBuffer.allocate(4);
            ch.read(head, 0);
            verifyMagic(head.array(), 0);
            byte[] footerBytes = new byte[footerLen];
            ByteBuffer fb = ByteBuffer.wrap(footerBytes);
            int n = ch.read(fb, footerStart);
            if (n != footerLen) throw new EOFException("Short footer read");
            return Util.readFileMetaData(new ByteArrayInputStream(footerBytes));
        }
    }

    public static void writeFileMetaDataTrailer(OutputStream out, FileMetaData meta) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Util.writeFileMetaData(meta, bos);
        byte[] footer = bos.toByteArray();
        out.write(footer);
        ByteBuffer len = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        len.putInt(footer.length);
        out.write(len.array());
        out.write(MAGIC);
    }

    public static byte[] readFully(Path path, long offset, int length) throws IOException {
        if (length < 0) throw new IOException("negative length");
        byte[] buf = new byte[length];
        if (length == 0) return buf;
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer bb = ByteBuffer.wrap(buf);
            int total = 0;
            while (total < length) {
                int n = ch.read(bb, offset + total);
                if (n < 0) throw new EOFException("EOF at offset " + (offset + total));
                total += n;
            }
        }
        return buf;
    }

    // ---- compression -------------------------------------------------------

    public static byte[] compress(CompressionCodec codec, byte[] raw) throws IOException {
        if (codec == null || codec == CompressionCodec.UNCOMPRESSED) return raw;
        switch (codec) {
            case SNAPPY:
                return Snappy.compress(raw);
            case ZSTD: {
                long max = Zstd.compressBound(raw.length);
                byte[] out = new byte[(int) max];
                long n = Zstd.compress(out, raw, 3);
                if (Zstd.isError(n)) throw new IOException("ZSTD compress: " + Zstd.getErrorName(n));
                return Arrays.copyOf(out, (int) n);
            }
            case GZIP: {
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                try (java.util.zip.GZIPOutputStream gz = new java.util.zip.GZIPOutputStream(bos)) {
                    gz.write(raw);
                }
                return bos.toByteArray();
            }
            default:
                throw new IOException("Unsupported compression: " + codec);
        }
    }

    public static byte[] decompress(CompressionCodec codec, byte[] compressed, int uncompressedSize)
            throws IOException {
        if (codec == null || codec == CompressionCodec.UNCOMPRESSED) {
            return compressed;
        }
        switch (codec) {
            case SNAPPY: {
                byte[] out = new byte[uncompressedSize];
                int n = Snappy.uncompress(compressed, 0, compressed.length, out, 0);
                if (n != uncompressedSize) {
                    throw new IOException("SNAPPY size mismatch: " + n + " vs " + uncompressedSize);
                }
                return out;
            }
            case ZSTD: {
                byte[] out = new byte[uncompressedSize];
                long n = Zstd.decompress(out, compressed);
                if (n != uncompressedSize) {
                    throw new IOException("ZSTD size mismatch: " + n + " vs " + uncompressedSize);
                }
                return out;
            }
            case GZIP: {
                ByteArrayOutputStream bos = new ByteArrayOutputStream(uncompressedSize);
                try (java.util.zip.GZIPInputStream gz =
                         new java.util.zip.GZIPInputStream(new ByteArrayInputStream(compressed))) {
                    byte[] buf = new byte[8192];
                    int r;
                    while ((r = gz.read(buf)) >= 0) bos.write(buf, 0, r);
                }
                return bos.toByteArray();
            }
            default:
                throw new IOException("Unsupported compression: " + codec);
        }
    }

    public static CompressionCodec toThrift(CompressionCodecName name) {
        if (name == null) return CompressionCodec.UNCOMPRESSED;
        switch (name) {
            case SNAPPY: return CompressionCodec.SNAPPY;
            case ZSTD: return CompressionCodec.ZSTD;
            case GZIP: return CompressionCodec.GZIP;
            case LZO: return CompressionCodec.LZO;
            case BROTLI: return CompressionCodec.BROTLI;
            case LZ4: return CompressionCodec.LZ4;
            case LZ4_RAW: return CompressionCodec.LZ4_RAW;
            case UNCOMPRESSED:
            default: return CompressionCodec.UNCOMPRESSED;
        }
    }

    // ---- encoding map ------------------------------------------------------

    public static Encoding fromThrift(org.apache.parquet.format.Encoding e) {
        if (e == null) return Encoding.PLAIN;
        switch (e) {
            case RLE: return Encoding.RLE;
            case BIT_PACKED: return Encoding.BIT_PACKED;
            case PLAIN_DICTIONARY: return Encoding.PLAIN_DICTIONARY;
            case DELTA_BINARY_PACKED: return Encoding.DELTA_BINARY_PACKED;
            case DELTA_LENGTH_BYTE_ARRAY: return Encoding.DELTA_LENGTH_BYTE_ARRAY;
            case DELTA_BYTE_ARRAY: return Encoding.DELTA_BYTE_ARRAY;
            case RLE_DICTIONARY: return Encoding.RLE_DICTIONARY;
            case BYTE_STREAM_SPLIT: return Encoding.BYTE_STREAM_SPLIT;
            case PLAIN:
            default: return Encoding.PLAIN;
        }
    }

    public static org.apache.parquet.format.Encoding toThrift(Encoding e) {
        if (e == null) return org.apache.parquet.format.Encoding.PLAIN;
        switch (e) {
            case RLE: return org.apache.parquet.format.Encoding.RLE;
            case BIT_PACKED: return org.apache.parquet.format.Encoding.BIT_PACKED;
            case PLAIN_DICTIONARY: return org.apache.parquet.format.Encoding.PLAIN_DICTIONARY;
            case DELTA_BINARY_PACKED: return org.apache.parquet.format.Encoding.DELTA_BINARY_PACKED;
            case DELTA_LENGTH_BYTE_ARRAY: return org.apache.parquet.format.Encoding.DELTA_LENGTH_BYTE_ARRAY;
            case DELTA_BYTE_ARRAY: return org.apache.parquet.format.Encoding.DELTA_BYTE_ARRAY;
            case RLE_DICTIONARY: return org.apache.parquet.format.Encoding.RLE_DICTIONARY;
            case BYTE_STREAM_SPLIT: return org.apache.parquet.format.Encoding.BYTE_STREAM_SPLIT;
            case PLAIN:
            default: return org.apache.parquet.format.Encoding.PLAIN;
        }
    }

    // ---- page header parse/write -------------------------------------------

    public static PageHeader readPageHeader(InputStream in) throws IOException {
        return Util.readPageHeader(in);
    }

    public static void writePageHeader(PageHeader header, OutputStream out) throws IOException {
        Util.writePageHeader(header, out);
    }

    public static PageHeader dataPageV1Header(int numValues, int uncompressed, int compressed,
                                             Encoding values, Encoding rl, Encoding dl) {
        DataPageHeader dph = new DataPageHeader(numValues, toThrift(values), toThrift(dl), toThrift(rl));
        PageHeader ph = new PageHeader(PageType.DATA_PAGE, uncompressed, compressed);
        ph.setData_page_header(dph);
        return ph;
    }

    public static PageHeader dictionaryPageHeader(int numValues, int uncompressed, int compressed,
                                                  Encoding encoding) {
        DictionaryPageHeader dph = new DictionaryPageHeader(numValues, toThrift(encoding));
        PageHeader ph = new PageHeader(PageType.DICTIONARY_PAGE, uncompressed, compressed);
        ph.setDictionary_page_header(dph);
        return ph;
    }

    public static DataPageV1 toDataPageV1(PageHeader ph, byte[] uncompressedBytes) {
        DataPageHeader dph = ph.getData_page_header();
        return new DataPageV1(
            org.apache.parquet.bytes.BytesInput.from(uncompressedBytes),
            dph.getNum_values(),
            ph.getUncompressed_page_size(),
            org.apache.parquet.column.statistics.Statistics.getStatsBasedOnType(PrimitiveTypeName.BINARY),
            fromThrift(dph.getRepetition_level_encoding()),
            fromThrift(dph.getDefinition_level_encoding()),
            fromThrift(dph.getEncoding()));
    }

    public static DictionaryPage toDictionaryPage(PageHeader ph, byte[] uncompressedBytes) {
        DictionaryPageHeader dph = ph.getDictionary_page_header();
        return new DictionaryPage(
            org.apache.parquet.bytes.BytesInput.from(uncompressedBytes),
            dph.getNum_values(),
            fromThrift(dph.getEncoding()));
    }

    // ---- schema MessageType ↔ thrift SchemaElement -------------------------

    public static List<SchemaElement> toThriftSchema(MessageType schema) {
        List<SchemaElement> out = new ArrayList<>();
        SchemaElement root = new SchemaElement(schema.getName());
        root.setNum_children(schema.getFieldCount());
        out.add(root);
        for (org.apache.parquet.schema.Type f : schema.getFields()) {
            addSchemaElements(out, f);
        }
        return out;
    }

    private static void addSchemaElements(List<SchemaElement> out, org.apache.parquet.schema.Type t) {
        SchemaElement se = new SchemaElement(t.getName());
        se.setRepetition_type(toThriftRep(t.getRepetition()));
        if (t.isPrimitive()) {
            PrimitiveType pt = t.asPrimitiveType();
            se.setType(toThriftType(pt.getPrimitiveTypeName()));
            if (pt.getTypeLength() > 0) se.setType_length(pt.getTypeLength());
            applyLogical(se, t.getLogicalTypeAnnotation(), t.getOriginalType());
            out.add(se);
        } else {
            GroupType gt = t.asGroupType();
            se.setNum_children(gt.getFieldCount());
            applyLogical(se, t.getLogicalTypeAnnotation(), t.getOriginalType());
            out.add(se);
            for (org.apache.parquet.schema.Type child : gt.getFields()) {
                addSchemaElements(out, child);
            }
        }
    }

    public static MessageType fromThriftSchema(List<SchemaElement> elements) {
        if (elements == null || elements.isEmpty()) {
            return new MessageType("root", List.of());
        }
        int[] idx = {0};
        SchemaElement root = elements.get(0);
        idx[0] = 1;
        List<org.apache.parquet.schema.Type> fields = new ArrayList<>();
        int children = root.isSetNum_children() ? root.getNum_children() : 0;
        for (int i = 0; i < children; i++) {
            fields.add(readType(elements, idx));
        }
        String name = root.getName() == null ? "root" : root.getName();
        return new MessageType(name, fields);
    }

    private static org.apache.parquet.schema.Type readType(List<SchemaElement> elements, int[] idx) {
        if (idx[0] >= elements.size()) {
            throw new IllegalArgumentException("Truncated thrift schema");
        }
        SchemaElement se = elements.get(idx[0]++);
        Repetition rep = fromThriftRep(se.getRepetition_type());
        if (se.isSetType()) {
            PrimitiveTypeName ptn = fromThriftType(se.getType());
            int len = se.isSetType_length() ? se.getType_length() : 0;
            Types.PrimitiveBuilder<PrimitiveType> b =
                Types.primitive(ptn, rep);
            if (len > 0) b.length(len);
            applyLogicalFromThrift(b, se);
            return b.named(se.getName());
        }
        int n = se.isSetNum_children() ? se.getNum_children() : 0;
        List<org.apache.parquet.schema.Type> children = new ArrayList<>(n);
        for (int i = 0; i < n; i++) children.add(readType(elements, idx));
        Types.GroupBuilder<GroupType> gb = Types.buildGroup(rep);
        if (se.isSetConverted_type()) {
            OriginalType ot = fromConverted(se.getConverted_type());
            if (ot != null) gb.as(ot);
        } else if (se.isSetLogicalType()) {
            LogicalTypeAnnotation lta = fromThriftLogical(se.getLogicalType());
            if (lta != null) gb.as(lta);
        }
        for (org.apache.parquet.schema.Type c : children) gb.addField(c);
        return gb.named(se.getName());
    }

    private static void applyLogical(SchemaElement se, LogicalTypeAnnotation lta, OriginalType ot) {
        if (lta instanceof LogicalTypeAnnotation.StringLogicalTypeAnnotation
            || ot == OriginalType.UTF8) {
            se.setConverted_type(ConvertedType.UTF8);
            se.setLogicalType(LogicalType.STRING(new StringType()));
        } else if (lta instanceof LogicalTypeAnnotation.ListLogicalTypeAnnotation
            || ot == OriginalType.LIST) {
            se.setConverted_type(ConvertedType.LIST);
        } else if (lta instanceof LogicalTypeAnnotation.MapLogicalTypeAnnotation
            || ot == OriginalType.MAP) {
            se.setConverted_type(ConvertedType.MAP);
        } else if (lta instanceof LogicalTypeAnnotation.EnumLogicalTypeAnnotation
            || ot == OriginalType.ENUM) {
            se.setConverted_type(ConvertedType.ENUM);
        } else if (lta instanceof LogicalTypeAnnotation.JsonLogicalTypeAnnotation
            || ot == OriginalType.JSON) {
            se.setConverted_type(ConvertedType.JSON);
        } else if (lta instanceof LogicalTypeAnnotation.DateLogicalTypeAnnotation
            || ot == OriginalType.DATE) {
            se.setConverted_type(ConvertedType.DATE);
        } else if (ot != null) {
            ConvertedType ct = toConverted(ot);
            if (ct != null) se.setConverted_type(ct);
        }
    }

    private static void applyLogicalFromThrift(Types.PrimitiveBuilder<PrimitiveType> b, SchemaElement se) {
        if (se.isSetLogicalType()) {
            LogicalTypeAnnotation lta = fromThriftLogical(se.getLogicalType());
            if (lta != null) {
                b.as(lta);
                return;
            }
        }
        if (se.isSetConverted_type()) {
            OriginalType ot = fromConverted(se.getConverted_type());
            if (ot != null) b.as(ot);
        }
    }

    private static LogicalTypeAnnotation fromThriftLogical(LogicalType lt) {
        if (lt == null) return null;
        if (lt.isSetSTRING()) return LogicalTypeAnnotation.stringType();
        if (lt.isSetLIST()) return LogicalTypeAnnotation.listType();
        if (lt.isSetMAP()) return LogicalTypeAnnotation.mapType();
        if (lt.isSetENUM()) return LogicalTypeAnnotation.enumType();
        if (lt.isSetJSON()) return LogicalTypeAnnotation.jsonType();
        if (lt.isSetDATE()) return LogicalTypeAnnotation.dateType();
        return null;
    }

    private static OriginalType fromConverted(ConvertedType ct) {
        if (ct == null) return null;
        switch (ct) {
            case UTF8: return OriginalType.UTF8;
            case LIST: return OriginalType.LIST;
            case MAP: return OriginalType.MAP;
            case MAP_KEY_VALUE: return OriginalType.MAP_KEY_VALUE;
            case ENUM: return OriginalType.ENUM;
            case DECIMAL: return OriginalType.DECIMAL;
            case DATE: return OriginalType.DATE;
            case TIME_MILLIS: return OriginalType.TIME_MILLIS;
            case TIME_MICROS: return OriginalType.TIME_MICROS;
            case TIMESTAMP_MILLIS: return OriginalType.TIMESTAMP_MILLIS;
            case TIMESTAMP_MICROS: return OriginalType.TIMESTAMP_MICROS;
            case JSON: return OriginalType.JSON;
            case BSON: return OriginalType.BSON;
            case INT_8: return OriginalType.INT_8;
            case INT_16: return OriginalType.INT_16;
            case INT_32: return OriginalType.INT_32;
            case INT_64: return OriginalType.INT_64;
            case UINT_8: return OriginalType.UINT_8;
            case UINT_16: return OriginalType.UINT_16;
            case UINT_32: return OriginalType.UINT_32;
            case UINT_64: return OriginalType.UINT_64;
            default: return null;
        }
    }

    private static ConvertedType toConverted(OriginalType ot) {
        if (ot == null) return null;
        try {
            return ConvertedType.valueOf(ot.name());
        } catch (Exception e) {
            return null;
        }
    }

    private static FieldRepetitionType toThriftRep(Repetition r) {
        if (r == null) return FieldRepetitionType.OPTIONAL;
        switch (r) {
            case REQUIRED: return FieldRepetitionType.REQUIRED;
            case REPEATED: return FieldRepetitionType.REPEATED;
            case OPTIONAL:
            default: return FieldRepetitionType.OPTIONAL;
        }
    }

    private static Repetition fromThriftRep(FieldRepetitionType r) {
        if (r == null) return Repetition.OPTIONAL;
        switch (r) {
            case REQUIRED: return Repetition.REQUIRED;
            case REPEATED: return Repetition.REPEATED;
            case OPTIONAL:
            default: return Repetition.OPTIONAL;
        }
    }

    /** Public wrapper for column-chunk thrift type mapping. */
    public static Type toThriftTypePublic(PrimitiveTypeName n) {
        return toThriftType(n);
    }

    private static Type toThriftType(PrimitiveTypeName n) {
        switch (n) {
            case BOOLEAN: return Type.BOOLEAN;
            case INT32: return Type.INT32;
            case INT64: return Type.INT64;
            case INT96: return Type.INT96;
            case FLOAT: return Type.FLOAT;
            case DOUBLE: return Type.DOUBLE;
            case FIXED_LEN_BYTE_ARRAY: return Type.FIXED_LEN_BYTE_ARRAY;
            case BINARY:
            default: return Type.BYTE_ARRAY;
        }
    }

    private static PrimitiveTypeName fromThriftType(Type t) {
        if (t == null) return PrimitiveTypeName.BINARY;
        switch (t) {
            case BOOLEAN: return PrimitiveTypeName.BOOLEAN;
            case INT32: return PrimitiveTypeName.INT32;
            case INT64: return PrimitiveTypeName.INT64;
            case INT96: return PrimitiveTypeName.INT96;
            case FLOAT: return PrimitiveTypeName.FLOAT;
            case DOUBLE: return PrimitiveTypeName.DOUBLE;
            case FIXED_LEN_BYTE_ARRAY: return PrimitiveTypeName.FIXED_LEN_BYTE_ARRAY;
            case BYTE_ARRAY:
            default: return PrimitiveTypeName.BINARY;
        }
    }

    public static String pathKey(List<String> path) {
        if (path == null || path.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < path.size(); i++) {
            if (i > 0) sb.append('.');
            sb.append(path.get(i));
        }
        return sb.toString();
    }

    public static String pathKey(String[] path) {
        if (path == null || path.length == 0) return "";
        return pathKey(Arrays.asList(path));
    }
}
