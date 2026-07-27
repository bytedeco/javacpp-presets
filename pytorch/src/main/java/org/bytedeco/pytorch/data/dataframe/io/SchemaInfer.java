package org.bytedeco.pytorch.data.dataframe.io;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.Type;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.Schema;
import org.bytedeco.pytorch.data.parquet.LocalParquetReader;

/**
 * Accurate multi-format schema inference for DataFrame loaders.
 *
 * <p>Combines extension detection ({@link FormatDetect}) with magic-byte
 * sniffing so misnamed files (e.g. {@code data.bin} that is actually parquet)
 * still load correctly. Nested LIST/MAP/STRUCT fields are preserved as
 * {@link Column.DType#LIST}/{@link Column.DType#MAP}/{@link Column.DType#STRUCT}.
 *
 * <pre>
 *   Schema s = SchemaInfer.infer("/path/to/valid.parquet");
 *   s.print(); // or DataFrame.read(path) which uses FormatDetect + this fallback
 * </pre>
 */
public final class SchemaInfer {
    private SchemaInfer() {}

    /** Infer schema without materializing all rows when possible. */
    public static Schema infer(String path) throws Exception {
        FormatDetect.Format fmt = FormatDetect.detect(path);
        if (fmt == FormatDetect.Format.UNKNOWN) {
            fmt = sniff(path);
        }
        return infer(path, fmt);
    }

    public static Schema infer(String path, FormatDetect.Format fmt) throws Exception {
        switch (fmt) {
            case PARQUET:
                return fromParquet(path);
            case CSV:
            case TSV:
            case JSON:
            case JSONL:
            case ARROW:
            case FEATHER:
            case PICKLE:
            case EXCEL:
            case HDF5:
            case AVRO:
            case ORC:
            case NPZ:
            case NPY:
            case SAFETENSORS:
            case GGUF:
                // Fall back: load (or head) via FormatDetect and take schema
                DataFrame df = FormatDetect.read(path);
                try {
                    return Schema.fromDataFrame(df);
                } finally {
                    try { df.close(); } catch (Exception ignored) {}
                }
            default:
                // last-chance sniff
                FormatDetect.Format sniffed = sniff(path);
                if (sniffed != FormatDetect.Format.UNKNOWN && sniffed != fmt) {
                    return infer(path, sniffed);
                }
                throw new IllegalArgumentException("Cannot infer schema for: " + path);
        }
    }

    /** Parquet schema only (no row materialization beyond footer). */
    public static Schema fromParquet(String path) throws IOException {
        try (LocalParquetReader r = LocalParquetReader.open(path)) {
            return fromParquetMessageType(r.getSchema());
        }
    }

    public static Schema fromParquetMessageType(MessageType mt) {
        Schema s = new Schema();
        for (Type field : mt.getFields()) {
            s.add(field.getName(), parquetTypeToDType(field));
        }
        return s;
    }

    /** Mirror of DataFrame.parquetTypeToDType for public schema peeking. */
    public static Column.DType parquetTypeToDType(Type ft) {
        if (ft.isPrimitive()) {
            org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName ptn =
                ft.asPrimitiveType().getPrimitiveTypeName();
            return switch (ptn) {
                case INT32 -> Column.DType.INT32;
                case INT64 -> Column.DType.INT64;
                case FLOAT -> Column.DType.FLOAT32;
                case DOUBLE -> Column.DType.FLOAT64;
                case BOOLEAN -> Column.DType.BOOLEAN;
                case BINARY, FIXED_LEN_BYTE_ARRAY -> {
                    var lta = ft.getLogicalTypeAnnotation();
                    if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.StringLogicalTypeAnnotation
                        || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.EnumLogicalTypeAnnotation
                        || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.JsonLogicalTypeAnnotation) {
                        yield Column.DType.STRING;
                    }
                    yield Column.DType.BINARY;
                }
                default -> Column.DType.STRING;
            };
        }
        var lta = ft.getLogicalTypeAnnotation();
        if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.ListLogicalTypeAnnotation
            || (!ft.isPrimitive()
                && ft.getRepetition() == Type.Repetition.REPEATED)) {
            Column.DType elem = listElementDType(ft);
            if (elem == Column.DType.FLOAT32 || elem == Column.DType.FLOAT64) {
                return Column.DType.VECTOR;
            }
            return Column.DType.LIST;
        }
        if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.MapLogicalTypeAnnotation
            || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.MapKeyValueTypeAnnotation) {
            return Column.DType.MAP;
        }
        return Column.DType.STRUCT;
    }

    private static Column.DType listElementDType(Type ft) {
        try {
            org.apache.parquet.schema.GroupType gt = ft.asGroupType();
            if (gt.getFieldCount() == 0) return Column.DType.STRING;
            Type mid = gt.getType(0);
            if (!mid.isPrimitive()) {
                org.apache.parquet.schema.GroupType midG = mid.asGroupType();
                if (midG.getFieldCount() > 0) {
                    Type elem = midG.getType(0);
                    if (elem.isPrimitive()) return parquetTypeToDType(elem);
                    return Column.DType.LIST;
                }
            } else {
                return parquetTypeToDType(mid);
            }
        } catch (Exception ignored) { /* fall through */ }
        return Column.DType.STRING;
    }

    /**
     * Magic-byte sniff when extension is missing/wrong.
     * Recognises parquet (PAR1), npy, npz (PK), arrow IPC, gzip-ish jsonl, etc.
     */
    public static FormatDetect.Format sniff(String path) {
        Path p = Path.of(path);
        if (!Files.isRegularFile(p)) return FormatDetect.Format.UNKNOWN;
        try (InputStream in = new BufferedInputStream(Files.newInputStream(p))) {
            byte[] head = in.readNBytes(16);
            if (head.length < 4) return FormatDetect.Format.UNKNOWN;
            // Parquet: "PAR1"
            if (head[0] == 'P' && head[1] == 'A' && head[2] == 'R' && head[3] == '1')
                return FormatDetect.Format.PARQUET;
            // NPY: \x93NUMPY
            if (head.length >= 6 && (head[0] & 0xFF) == 0x93
                && head[1] == 'N' && head[2] == 'U' && head[3] == 'M'
                && head[4] == 'P' && head[5] == 'Y')
                return FormatDetect.Format.NPY;
            // ZIP-based: npz, xlsx, orc sometimes
            if (head[0] == 'P' && head[1] == 'K') {
                String lower = path.toLowerCase(Locale.ROOT);
                if (lower.endsWith(".npz")) return FormatDetect.Format.NPZ;
                if (lower.endsWith(".xlsx") || lower.endsWith(".xlsm")) return FormatDetect.Format.EXCEL;
                // default zip → try npz
                return FormatDetect.Format.NPZ;
            }
            // Arrow IPC magic "ARROW1" or feather V1 "FEA1"
            String ascii = new String(head, 0, Math.min(6, head.length), StandardCharsets.US_ASCII);
            if (ascii.startsWith("ARROW1")) return FormatDetect.Format.ARROW;
            if (ascii.startsWith("FEA1")) return FormatDetect.Format.FEATHER;
            // Avro Object Container File: Obj\x01
            if (head[0] == 'O' && head[1] == 'b' && head[2] == 'j' && head[3] == 0x01)
                return FormatDetect.Format.AVRO;
            // ORC: "ORC"
            if (head[0] == 'O' && head[1] == 'R' && head[2] == 'C')
                return FormatDetect.Format.ORC;
            // HDF5: \x89HDF
            if ((head[0] & 0xFF) == 0x89 && head[1] == 'H' && head[2] == 'D' && head[3] == 'F')
                return FormatDetect.Format.HDF5;
            // JSON start
            int i = 0;
            while (i < head.length && Character.isWhitespace((char) head[i])) i++;
            if (i < head.length && (head[i] == '{' || head[i] == '['))
                return FormatDetect.Format.JSON;
            // GGUF
            if (ascii.startsWith("GGUF")) return FormatDetect.Format.GGUF;
            // Safetensors is JSON header length LE u64 then JSON — hard to sniff; leave UNKNOWN
            return FormatDetect.Format.UNKNOWN;
        } catch (IOException e) {
            return FormatDetect.Format.UNKNOWN;
        }
    }

    /** Human-readable schema dump (Spark-style). */
    public static void print(Schema schema) {
        System.out.println("root");
        List<String> names = schema.fieldNames();
        List<Column.DType> types = schema.fieldTypes();
        for (int i = 0; i < names.size(); i++) {
            System.out.printf(" |-- %s: %s%n", names.get(i), types.get(i));
        }
    }

    /** Describe nested parquet schema with logical types for debugging. */
    public static Map<String, String> describeParquet(String path) throws IOException {
        Map<String, String> out = new LinkedHashMap<>();
        try (LocalParquetReader r = LocalParquetReader.open(path)) {
            MessageType mt = r.getSchema();
            for (Type f : mt.getFields()) {
                out.put(f.getName(), f.toString().replace('\n', ' '));
            }
        }
        return out;
    }

    /** List field names only. */
    public static List<String> fieldNames(String path) throws Exception {
        return new ArrayList<>(infer(path).fieldNames());
    }
}
