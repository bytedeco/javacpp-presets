package org.bytedeco.pytorch.data.arrow;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.dataframe.io.ComplexCellCodec;

/**
 * Maps between DataFrame {@link Column.DType} and Apache Arrow field types.
 *
 * <p>Complex dtypes use native nested Arrow types where possible:
 * <ul>
 *   <li>{@code LIST} → {@code list&lt;item&gt;} (item is Utf8 for heterogeneous / nested JSON text)</li>
 *   <li>{@code VECTOR}/{@code EMBEDDING} → {@code FixedSizeList&lt;float32&gt;(dim)} when dim known,
 *       else {@code list&lt;float32&gt;}</li>
 *   <li>{@code MAP} → {@code map&lt;utf8, utf8&gt;} (values JSON-encoded when nested)</li>
 *   <li>{@code STRUCT} → native Arrow Struct when key set is stable; else Utf8 JSON text</li>
 *   <li>{@code JSON} → Utf8 (JSON text)</li>
 *   <li>{@code BINARY} → Binary</li>
 * </ul>
 */
public final class ArrowSchemaMapper {
    private ArrowSchemaMapper() {}

    public static Field toField(String name, Column.DType dtype) {
        return toField(name, dtype, -1, null);
    }

    /**
     * @param fixedDim vector dimension for VECTOR/EMBEDDING; &lt;=0 means variable list&lt;f32&gt;
     * @param structKeys ordered child names for STRUCT (null → Utf8 JSON fallback)
     */
    public static Field toField(String name, Column.DType dtype, int fixedDim, List<String> structKeys) {
        return switch (dtype) {
            case INT32 -> primitive(name, new ArrowType.Int(32, true));
            case INT64, DURATION -> primitive(name, new ArrowType.Int(64, true));
            case FLOAT32 -> primitive(name, new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE));
            case FLOAT64 -> primitive(name, new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE));
            case BOOLEAN -> primitive(name, new ArrowType.Bool());
            case DATE -> primitive(name, new ArrowType.Date(DateUnit.DAY));
            case DATETIME -> primitive(name, new ArrowType.Timestamp(TimeUnit.MILLISECOND, null));
            case TIME -> primitive(name, new ArrowType.Time(TimeUnit.MILLISECOND, 32));
            case BINARY -> primitive(name, new ArrowType.Binary());
            case VECTOR, EMBEDDING -> {
                Field item = primitive("item",
                    new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE));
                if (fixedDim > 0) {
                    yield fixedSizeListOf(name, fixedDim, item);
                }
                yield listOf(name, item);
            }
            case LIST -> listOf(name, primitive("item", new ArrowType.Utf8()));
            case MAP -> mapOf(name,
                primitive("key", new ArrowType.Utf8()),
                primitive("value", new ArrowType.Utf8()));
            case STRUCT -> {
                if (structKeys != null && !structKeys.isEmpty()) {
                    List<Field> children = new ArrayList<>(structKeys.size());
                    for (String k : structKeys) {
                        children.add(primitive(k, new ArrowType.Utf8()));
                    }
                    yield structOf(name, children);
                }
                yield primitive(name, new ArrowType.Utf8());
            }
            case JSON, STRING, TENSOR, IMAGE, AUDIO, VIDEO, GRAPH, POINT_CLOUD ->
                primitive(name, new ArrowType.Utf8());
        };
    }

    /**
     * Infer Arrow field from a live column (uses sampling for VECTOR dim / STRUCT keys).
     */
    public static Field toField(Column col) {
        Column.DType dt = col.dtype();
        if (dt == Column.DType.VECTOR || dt == Column.DType.EMBEDDING) {
            int dim = inferVectorDim(col);
            return toField(col.name(), dt, dim, null);
        }
        if (dt == Column.DType.STRUCT) {
            List<String> keys = inferStructKeys(col);
            return toField(col.name(), dt, -1, keys);
        }
        if (dt == Column.DType.MAP) {
            return toField(col.name(), dt, -1, null);
        }
        return toField(col.name(), dt);
    }

    public static Column.DType fromField(Field field) {
        ArrowType type = field.getType();
        if (type instanceof ArrowType.Int intType) {
            return intType.getBitWidth() <= 32 ? Column.DType.INT32 : Column.DType.INT64;
        }
        if (type instanceof ArrowType.FloatingPoint fp) {
            return fp.getPrecision() == FloatingPointPrecision.SINGLE
                ? Column.DType.FLOAT32 : Column.DType.FLOAT64;
        }
        if (type instanceof ArrowType.Bool) return Column.DType.BOOLEAN;
        if (type instanceof ArrowType.Utf8 || type instanceof ArrowType.LargeUtf8) {
            return Column.DType.STRING;
        }
        if (type instanceof ArrowType.Binary || type instanceof ArrowType.LargeBinary) {
            return Column.DType.BINARY;
        }
        if (type instanceof ArrowType.Date) return Column.DType.DATE;
        if (type instanceof ArrowType.Timestamp) return Column.DType.DATETIME;
        if (type instanceof ArrowType.Time) return Column.DType.TIME;
        if (type instanceof ArrowType.Duration) return Column.DType.DURATION;
        if (type instanceof ArrowType.Decimal) return Column.DType.FLOAT64;
        if (type instanceof ArrowType.List || type instanceof ArrowType.LargeList
            || type instanceof ArrowType.FixedSizeList) {
            // float/double element → VECTOR; else LIST
            List<Field> children = field.getChildren();
            if (children != null && !children.isEmpty()) {
                Column.DType elem = fromField(children.get(0));
                if (elem == Column.DType.FLOAT32 || elem == Column.DType.FLOAT64) {
                    return Column.DType.VECTOR;
                }
            }
            return Column.DType.LIST;
        }
        if (type instanceof ArrowType.Map) {
            return Column.DType.MAP;
        }
        if (type instanceof ArrowType.Struct) {
            return Column.DType.STRUCT;
        }
        return Column.DType.STRING;
    }

    /**
     * Sample non-null cells and return a consistent vector dim, or -1 if unknown / mixed.
     */
    public static int inferVectorDim(Column col) {
        if (col == null) return -1;
        int dim = -1;
        int checked = 0;
        int n = col.size();
        for (int i = 0; i < n && checked < 64; i++) {
            Object v = col.get(i);
            if (v == null) continue;
            int d = vectorLen(v);
            if (d <= 0) continue;
            if (dim < 0) dim = d;
            else if (dim != d) return -1; // mixed dims → variable list
            checked++;
        }
        return dim;
    }

    /**
     * Collect stable union of map keys from STRUCT cells (first 64 non-null).
     * Returns null if no map-like cells found.
     */
    public static List<String> inferStructKeys(Column col) {
        if (col == null) return null;
        Set<String> keys = new LinkedHashSet<>();
        int checked = 0;
        int n = col.size();
        for (int i = 0; i < n && checked < 64; i++) {
            Object v = col.get(i);
            if (v == null) continue;
            Map<String, Object> map = ComplexCellCodec.asStringMap(v);
            if (map == null) continue;
            keys.addAll(map.keySet());
            checked++;
        }
        if (keys.isEmpty()) return null;
        return new ArrayList<>(keys);
    }

    private static int vectorLen(Object v) {
        if (v instanceof EmbeddingData ed) return ed.getDimension();
        if (v instanceof float[] f) return f.length;
        if (v instanceof double[] d) return d.length;
        if (v instanceof List<?> list) {
            // only count if numeric-looking
            if (list.isEmpty()) return 0;
            Object first = list.get(0);
            if (first instanceof Number) return list.size();
            return -1;
        }
        float[] coerced = null;
        try {
            Object dens = ComplexCellCodec.coerceComplex(v, Column.DType.VECTOR);
            if (dens instanceof float[] f) return f.length;
        } catch (Exception ignored) {
            // fall through
        }
        return coerced == null ? -1 : coerced.length;
    }

    private static Field primitive(String name, ArrowType type) {
        return new Field(name, FieldType.nullable(type), null);
    }

    private static Field listOf(String name, Field item) {
        return new Field(name, FieldType.nullable(new ArrowType.List()),
            Collections.singletonList(item));
    }

    private static Field fixedSizeListOf(String name, int dim, Field item) {
        return new Field(name, FieldType.nullable(new ArrowType.FixedSizeList(dim)),
            Collections.singletonList(item));
    }

    private static Field structOf(String name, List<Field> children) {
        return new Field(name, FieldType.nullable(new ArrowType.Struct()), children);
    }

    /**
     * Arrow Map type is a list of structs with key/value children.
     * Field layout: map&lt;entries: struct&lt;key, value&gt;&gt;
     */
    private static Field mapOf(String name, Field key, Field value) {
        // keys are non-nullable in Arrow map convention
        Field keyNN = new Field(key.getName(),
            new FieldType(false, key.getType(), null), key.getChildren());
        Field entries = new Field("entries",
            FieldType.notNullable(new ArrowType.Struct()),
            List.of(keyNN, value));
        return new Field(name, FieldType.nullable(new ArrowType.Map(false)),
            Collections.singletonList(entries));
    }
}
