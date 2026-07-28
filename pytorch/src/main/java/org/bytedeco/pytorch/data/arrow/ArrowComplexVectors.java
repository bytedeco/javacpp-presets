package org.bytedeco.pytorch.data.arrow;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeMilliVector;
import org.apache.arrow.vector.TimeStampMilliVector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.MapVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.complex.impl.UnionMapWriter;
import org.bytedeco.pytorch.dataframe.ArrowStorage;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.dataframe.io.ComplexCellCodec;

/**
 * Shared fill / read helpers for Arrow nested vectors used by
 * {@link LocalArrowIpcWriter}, {@link ArrowBridge} and {@link ArrowStorage}.
 */
public final class ArrowComplexVectors {
    private ArrowComplexVectors() {}

    /** Fill any field vector (primitive or nested) from a DataFrame column. */
    public static void fillVector(FieldVector vec, Column col, int n) {
        vec.setInitialCapacity(n);
        vec.allocateNew();
        if (vec instanceof IntVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else v.setSafe(i, ((Number) val).intValue());
            }
            v.setValueCount(n);
        } else if (vec instanceof BigIntVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else if (val instanceof Duration d) v.setSafe(i, d.toMillis());
                else v.setSafe(i, ((Number) val).longValue());
            }
            v.setValueCount(n);
        } else if (vec instanceof Float4Vector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else v.setSafe(i, ((Number) val).floatValue());
            }
            v.setValueCount(n);
        } else if (vec instanceof Float8Vector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else v.setSafe(i, ((Number) val).doubleValue());
            }
            v.setValueCount(n);
        } else if (vec instanceof BitVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else {
                    boolean b = Boolean.TRUE.equals(val)
                        || (val instanceof Number && ((Number) val).intValue() != 0);
                    v.setSafe(i, b ? 1 : 0);
                }
            }
            v.setValueCount(n);
        } else if (vec instanceof DateDayVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else if (val instanceof LocalDate ld) v.setSafe(i, (int) ld.toEpochDay());
                else if (val instanceof Number num) v.setSafe(i, num.intValue());
                else v.setSafe(i, (int) LocalDate.parse(val.toString()).toEpochDay());
            }
            v.setValueCount(n);
        } else if (vec instanceof TimeStampMilliVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else v.setSafe(i, toEpochMilli(val));
            }
            v.setValueCount(n);
        } else if (vec instanceof TimeMilliVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else if (val instanceof LocalTime lt)
                    v.setSafe(i, (int) (lt.toNanoOfDay() / 1_000_000L));
                else if (val instanceof Number num) v.setSafe(i, num.intValue());
                else v.setSafe(i, (int) (LocalTime.parse(val.toString()).toNanoOfDay() / 1_000_000L));
            }
            v.setValueCount(n);
        } else if (vec instanceof VarBinaryVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else if (val instanceof byte[] b) v.setSafe(i, b);
                else v.setSafe(i, String.valueOf(val).getBytes(StandardCharsets.UTF_8));
            }
            v.setValueCount(n);
        } else if (vec instanceof MapVector mv) {
            fillMapVector(mv, col, n);
        } else if (vec instanceof FixedSizeListVector fsl) {
            fillFixedSizeListVector(fsl, col, n);
        } else if (vec instanceof ListVector lv) {
            fillListVector(lv, col, n);
        } else if (vec instanceof StructVector sv) {
            // STRUCT as map-of-fields; multi-child writes by name when possible
            fillStructAsJsonText(sv, col, n);
        } else if (vec instanceof VarCharVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else {
                    String text = complexOrString(val, col.dtype());
                    v.setSafe(i, text.getBytes(StandardCharsets.UTF_8));
                }
            }
            v.setValueCount(n);
        } else {
            for (int i = 0; i < n; i++) vec.setNull(i);
            vec.setValueCount(n);
        }
    }

    /** Read a cell, densifying list&lt;float&gt; → float[] for VECTOR/EMBEDDING. */
    public static Object readValue(FieldVector vec, int index, Column.DType dtype) {
        if (vec.isNull(index)) return null;

        if (vec instanceof IntVector v) return v.get(index);
        if (vec instanceof BigIntVector v) return v.get(index);
        if (vec instanceof Float4Vector v) return v.get(index);
        if (vec instanceof Float8Vector v) return v.get(index);
        if (vec instanceof BitVector v) return v.get(index) == 1;
        if (vec instanceof VarCharVector v) {
            byte[] b = v.get(index);
            if (b == null) return null;
            String s = new String(b, StandardCharsets.UTF_8);
            if (ComplexCellCodec.isComplex(dtype) || ComplexCellCodec.isListLike(dtype)
                || ComplexCellCodec.isMapLike(dtype)) {
                return ComplexCellCodec.decodeText(s, dtype);
            }
            return s;
        }
        if (vec instanceof VarBinaryVector v) {
            return v.get(index);
        }
        if (vec instanceof DateDayVector v) {
            return LocalDate.ofEpochDay(v.get(index));
        }
        if (vec instanceof TimeStampMilliVector v) {
            return Instant.ofEpochMilli(v.get(index));
        }
        if (vec instanceof TimeMilliVector v) {
            return LocalTime.ofNanoOfDay(v.get(index) * 1_000_000L);
        }
        if (vec instanceof MapVector mv) {
            return readMap(mv, index);
        }
        if (vec instanceof FixedSizeListVector fsl) {
            Object list = readFixedSizeList(fsl, index);
            if (dtype == Column.DType.VECTOR || dtype == Column.DType.EMBEDDING) {
                return ComplexCellCodec.coerceComplex(list, Column.DType.VECTOR);
            }
            if (dtype == Column.DType.LIST) {
                return ComplexCellCodec.coerceComplex(list, Column.DType.LIST);
            }
            return densifyIfNumeric(list);
        }
        if (vec instanceof ListVector lv) {
            Object list = readList(lv, index);
            if (dtype == Column.DType.VECTOR || dtype == Column.DType.EMBEDDING) {
                return ComplexCellCodec.coerceComplex(list, Column.DType.VECTOR);
            }
            if (dtype == Column.DType.LIST) {
                return ComplexCellCodec.coerceComplex(list, Column.DType.LIST);
            }
            // auto: float lists densify to float[]
            return densifyIfNumeric(list);
        }
        if (vec instanceof StructVector sv) {
            return readStruct(sv, index);
        }

        Object o = vec.getObject(index);
        if (o == null) return null;
        if (dtype == Column.DType.VECTOR || dtype == Column.DType.EMBEDDING) {
            return ComplexCellCodec.coerceComplex(o, Column.DType.VECTOR);
        }
        if (dtype == Column.DType.LIST) {
            return ComplexCellCodec.coerceComplex(o, Column.DType.LIST);
        }
        if (dtype == Column.DType.MAP || dtype == Column.DType.STRUCT) {
            return ComplexCellCodec.coerceComplex(o, dtype);
        }
        return o;
    }

    // ---- list write ---------------------------------------------------------

    /**
     * Write VECTOR/EMBEDDING cells into a FixedSizeList&lt;float32&gt;(dim) column.
     * Null / short vectors are padded with 0; longer vectors are truncated to dim.
     */
    private static void fillFixedSizeListVector(FixedSizeListVector fsl, Column col, int n) {
        int dim = fsl.getListSize();
        FieldVector data = fsl.getDataVector();
        fsl.allocateNew();
        if (data instanceof Float4Vector) {
            ((Float4Vector) data).allocateNew(Math.max(n * dim, 1));
        } else if (data instanceof Float8Vector) {
            ((Float8Vector) data).allocateNew(Math.max(n * dim, 1));
        } else {
            data.allocateNew();
        }
        int elemPos = 0;
        for (int i = 0; i < n; i++) {
            Object val = col.get(i);
            if (val == null) {
                fsl.setNull(i);
                // FixedSizeList still advances the child slot for null rows in some Arrow versions;
                // pad with nulls to keep child length = n * dim.
                for (int d = 0; d < dim; d++) {
                    data.setNull(elemPos++);
                }
                continue;
            }
            fsl.setNotNull(i);
            fsl.startNewValue(i);
            float[] vec = toFloatArray(val, dim);
            for (int d = 0; d < dim; d++) {
                float f = vec != null && d < vec.length ? vec[d] : 0f;
                if (data instanceof Float4Vector v) {
                    v.setSafe(elemPos, f);
                } else if (data instanceof Float8Vector v) {
                    v.setSafe(elemPos, f);
                } else if (data instanceof IntVector v) {
                    v.setSafe(elemPos, (int) f);
                } else {
                    data.setNull(elemPos);
                }
                elemPos++;
            }
        }
        data.setValueCount(elemPos);
        fsl.setValueCount(n);
    }

    private static float[] toFloatArray(Object val, int dim) {
        if (val instanceof EmbeddingData ed) return ed.getVector();
        if (val instanceof float[] f) return f;
        if (val instanceof double[] d) {
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            return f;
        }
        List<Object> elems = ComplexCellCodec.asObjectList(val);
        if (elems == null) {
            Object coerced = ComplexCellCodec.coerceComplex(val, Column.DType.VECTOR);
            if (coerced instanceof float[] f) return f;
            return null;
        }
        float[] f = new float[elems.size()];
        for (int i = 0; i < elems.size(); i++) {
            Object e = elems.get(i);
            f[i] = e instanceof Number ? ((Number) e).floatValue() : 0f;
        }
        return f;
    }

    private static void fillListVector(ListVector lv, Column col, int n) {
        // Prefer low-level startNewValue/endValue — UnionListWriter offset buffers are easy to under-allocate.
        FieldVector data = lv.getDataVector();
        int elemPos = 0;
        // Pre-size data vector roughly
        int estElems = Math.max(n * 4, 16);
        if (data instanceof Float4Vector) ((Float4Vector) data).allocateNew(estElems);
        else if (data instanceof Float8Vector) ((Float8Vector) data).allocateNew(estElems);
        else if (data instanceof IntVector) ((IntVector) data).allocateNew(estElems);
        else if (data instanceof BigIntVector) ((BigIntVector) data).allocateNew(estElems);
        else if (data instanceof BitVector) ((BitVector) data).allocateNew(estElems);
        else if (data instanceof VarCharVector) ((VarCharVector) data).allocateNew();
        else data.allocateNew();

        for (int i = 0; i < n; i++) {
            Object val = col.get(i);
            if (val == null) {
                lv.setNull(i);
                continue;
            }
            List<Object> elems = ComplexCellCodec.asObjectList(val);
            if (elems == null) {
                lv.setNull(i);
                continue;
            }
            lv.startNewValue(i);
            for (Object e : elems) {
                if (e == null) {
                    data.setNull(elemPos);
                } else if (data instanceof Float4Vector v) {
                    float f = e instanceof Number ? ((Number) e).floatValue()
                        : Float.parseFloat(String.valueOf(e));
                    v.setSafe(elemPos, f);
                } else if (data instanceof Float8Vector v) {
                    double d = e instanceof Number ? ((Number) e).doubleValue()
                        : Double.parseDouble(String.valueOf(e));
                    v.setSafe(elemPos, d);
                } else if (data instanceof IntVector v) {
                    v.setSafe(elemPos, e instanceof Number ? ((Number) e).intValue()
                        : Integer.parseInt(String.valueOf(e)));
                } else if (data instanceof BigIntVector v) {
                    v.setSafe(elemPos, e instanceof Number ? ((Number) e).longValue()
                        : Long.parseLong(String.valueOf(e)));
                } else if (data instanceof BitVector v) {
                    boolean b = e instanceof Boolean ? (Boolean) e
                        : Boolean.parseBoolean(String.valueOf(e));
                    v.setSafe(elemPos, b ? 1 : 0);
                } else if (data instanceof VarCharVector v) {
                    String s;
                    if (e instanceof Map || e instanceof List || e.getClass().isArray()) {
                        s = ComplexCellCodec.encodeText(e);
                    } else {
                        s = String.valueOf(e);
                    }
                    v.setSafe(elemPos, s.getBytes(StandardCharsets.UTF_8));
                } else {
                    // unsupported element vector — skip
                    data.setNull(elemPos);
                }
                elemPos++;
            }
            lv.endValue(i, elems.size());
        }
        data.setValueCount(elemPos);
        lv.setValueCount(n);
    }
    private static void fillMapVector(MapVector mv, Column col, int n) {
        UnionMapWriter writer = mv.getWriter();
        writer.allocate();
        for (int i = 0; i < n; i++) {
            Object val = col.get(i);
            writer.setPosition(i);
            if (val == null) {
                writer.writeNull();
                continue;
            }
            Map<String, Object> map = ComplexCellCodec.asStringMap(val);
            if (map == null) {
                writer.writeNull();
                continue;
            }
            writer.startMap();
            for (Map.Entry<String, Object> e : map.entrySet()) {
                writer.startEntry();
                // UnionMapWriter.key()/value() re-position the writer for that side
                writer.key().writeVarChar(e.getKey() == null ? "" : e.getKey());
                Object v = e.getValue();
                if (v == null) {
                    writer.value().writeNull();
                } else if (v instanceof Map || v instanceof List || v.getClass().isArray()) {
                    writer.value().writeVarChar(ComplexCellCodec.encodeText(v));
                } else {
                    writer.value().writeVarChar(String.valueOf(v));
                }
                writer.endEntry();
            }
            writer.endMap();
        }
        writer.setValueCount(n);
        mv.setValueCount(n);
    }

    /** STRUCT columns in flat DF schema: encode whole cell as JSON into first Utf8 child if present. */
    private static void fillStructAsJsonText(StructVector sv, Column col, int n) {
        // Prefer treating STRUCT cells as map → write children by name when possible
        List<FieldVector> children = sv.getChildrenFromFields();
        if (children == null || children.isEmpty()) {
            for (int i = 0; i < n; i++) sv.setNull(i);
            sv.setValueCount(n);
            return;
        }
        // If single utf8 child named "json" or first child is utf8 — write JSON text
        FieldVector first = children.get(0);
        if (children.size() == 1 && first instanceof VarCharVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) {
                    sv.setNull(i);
                } else {
                    sv.setIndexDefined(i);
                    v.setSafe(i, ComplexCellCodec.encodeText(val).getBytes(StandardCharsets.UTF_8));
                }
            }
            first.setValueCount(n);
            sv.setValueCount(n);
            return;
        }
        // Multi-child struct: map keys → children
        for (FieldVector child : children) {
            child.setInitialCapacity(n);
            child.allocateNew();
        }
        for (int i = 0; i < n; i++) {
            Object val = col.get(i);
            if (val == null) {
                sv.setNull(i);
                for (FieldVector child : children) child.setNull(i);
                continue;
            }
            sv.setIndexDefined(i);
            Map<String, Object> map = ComplexCellCodec.asStringMap(val);
            for (FieldVector child : children) {
                Object cv = map == null ? null : map.get(child.getName());
                if (cv == null) child.setNull(i);
                else if (child instanceof VarCharVector v) {
                    String s = (cv instanceof Map || cv instanceof List || cv.getClass().isArray())
                        ? ComplexCellCodec.encodeText(cv) : String.valueOf(cv);
                    v.setSafe(i, s.getBytes(StandardCharsets.UTF_8));
                } else if (child instanceof IntVector v && cv instanceof Number) {
                    v.setSafe(i, ((Number) cv).intValue());
                } else if (child instanceof BigIntVector v && cv instanceof Number) {
                    v.setSafe(i, ((Number) cv).longValue());
                } else if (child instanceof Float4Vector v && cv instanceof Number) {
                    v.setSafe(i, ((Number) cv).floatValue());
                } else if (child instanceof Float8Vector v && cv instanceof Number) {
                    v.setSafe(i, ((Number) cv).doubleValue());
                } else if (child instanceof BitVector v) {
                    boolean b = cv instanceof Boolean ? (Boolean) cv
                        : Boolean.parseBoolean(String.valueOf(cv));
                    v.setSafe(i, b ? 1 : 0);
                } else if (child instanceof VarCharVector v) {
                    v.setSafe(i, String.valueOf(cv).getBytes(StandardCharsets.UTF_8));
                } else {
                    child.setNull(i);
                }
            }
        }
        for (FieldVector child : children) child.setValueCount(n);
        sv.setValueCount(n);
    }

    // ---- nested read --------------------------------------------------------

    private static Object readFixedSizeList(FixedSizeListVector fsl, int index) {
        if (fsl.isNull(index)) return null;
        Object obj = fsl.getObject(index);
        if (obj == null) return null;
        if (obj instanceof List) {
            List<?> src = (List<?>) obj;
            List<Object> out = new ArrayList<>(src.size());
            for (Object o : src) {
                out.add(unwrapArrowScalar(o));
            }
            return out;
        }
        return obj;
    }

    private static Object readList(ListVector lv, int index) {
        Object obj = lv.getObject(index);
        if (obj == null) return null;
        if (obj instanceof List) {
            // unwrap JsonString / Text / numbers from Arrow
            List<?> src = (List<?>) obj;
            List<Object> out = new ArrayList<>(src.size());
            for (Object o : src) {
                out.add(unwrapArrowScalar(o));
            }
            return out;
        }
        return obj;
    }

    private static Map<String, Object> readMap(MapVector mv, int index) {
        Object obj = mv.getObject(index);
        Map<String, Object> out = new LinkedHashMap<>();
        if (obj == null) return out;
        if (obj instanceof List) {
            // Arrow Map getObject often returns List of entry structs / maps
            for (Object entry : (List<?>) obj) {
                if (entry instanceof Map) {
                    Map<?, ?> em = (Map<?, ?>) entry;
                    Object k = em.containsKey("key") ? em.get("key")
                        : (em.isEmpty() ? null : em.values().iterator().next());
                    Object v = em.containsKey("value") ? em.get("value") : null;
                    // some implementations use entry as 2-element
                    if (k == null && em.size() >= 1) {
                        Object[] vals = em.values().toArray();
                        k = vals[0];
                        v = vals.length > 1 ? vals[1] : null;
                    }
                    out.put(String.valueOf(unwrapArrowScalar(k)), unwrapArrowScalar(v));
                } else if (entry instanceof List && ((List<?>) entry).size() >= 2) {
                    List<?> el = (List<?>) entry;
                    out.put(String.valueOf(unwrapArrowScalar(el.get(0))), unwrapArrowScalar(el.get(1)));
                }
            }
            return out;
        }
        if (obj instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) obj;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                out.put(String.valueOf(e.getKey()), unwrapArrowScalar(e.getValue()));
            }
        }
        return out;
    }

    private static Map<String, Object> readStruct(StructVector sv, int index) {
        Map<String, Object> out = new LinkedHashMap<>();
        List<FieldVector> children = sv.getChildrenFromFields();
        if (children == null) return out;
        for (FieldVector child : children) {
            Object v = child.isNull(index) ? null
                : readValue(child, index, ArrowSchemaMapper.fromField(child.getField()));
            out.put(child.getName(), v);
        }
        return out;
    }

    private static Object unwrapArrowScalar(Object o) {
        if (o == null) return null;
        // Arrow Text / JsonStringHolder etc.
        if (o instanceof org.apache.arrow.vector.util.Text t) {
            return t.toString();
        }
        if (o instanceof byte[]) {
            return new String((byte[]) o, StandardCharsets.UTF_8);
        }
        return o;
    }

    private static Object densifyIfNumeric(Object list) {
        if (!(list instanceof List)) return list;
        return ComplexCellCodec.densifyList((List<?>) list);
    }

    private static String complexOrString(Object val, Column.DType dtype) {
        if (val == null) return null;
        if (ComplexCellCodec.isComplex(dtype) || ComplexCellCodec.isListLike(dtype)
            || ComplexCellCodec.isMapLike(dtype)
            || val instanceof Map || val instanceof List
            || val instanceof float[] || val instanceof double[]
            || val instanceof int[] || val instanceof long[]) {
            return ComplexCellCodec.encodeText(val);
        }
        return String.valueOf(val);
    }

    private static long toEpochMilli(Object val) {
        if (val instanceof Instant in) return in.toEpochMilli();
        if (val instanceof LocalDateTime ldt) return ldt.toInstant(ZoneOffset.UTC).toEpochMilli();
        if (val instanceof ZonedDateTime zdt) return zdt.toInstant().toEpochMilli();
        if (val instanceof LocalDate ld) return ld.atStartOfDay().toInstant(ZoneOffset.UTC).toEpochMilli();
        if (val instanceof Number num) return num.longValue();
        return Instant.parse(val.toString()).toEpochMilli();
    }
}
