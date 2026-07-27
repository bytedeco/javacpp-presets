package org.bytedeco.pytorch.data.orc;

import org.apache.orc.OrcProto;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.util.ArrayList;
import java.util.List;

/**
 * Map between DataFrame column dtypes and ORC {@link OrcProto.Type} trees.
 *
 * <p>Supports flat STRUCT of primitives plus one-level LIST / VECTOR / EMBEDDING
 * (ORC {@code list&lt;T&gt;} with primitive element). No MAP/UNION/nested-LIST in this
 * pass.
 */
final class OrcTypeMapper {
    private OrcTypeMapper() {}

    /**
     * One top-level field under the root STRUCT.
     * For LIST/VECTOR/EMBEDDING, {@link #elementKind} is set and the ORC type
     * tree has an extra child column id for the element type.
     */
    static final class Field {
        final String name;
        final Column.DType dtype;
        final OrcProto.Type.Kind kind;
        /** Column id of this field in the ORC type tree. */
        final int columnId;
        /** Element kind for LIST/VECTOR/EMBEDDING; null for scalars. */
        final OrcProto.Type.Kind elementKind;
        /** Element column id (subtype of list); -1 for scalars. */
        final int elementColumnId;
        /** Preferred DataFrame element dtype for densifying arrays on read. */
        final Column.DType elementDtype;

        Field(String name, Column.DType dtype, OrcProto.Type.Kind kind, int columnId,
              OrcProto.Type.Kind elementKind, int elementColumnId, Column.DType elementDtype) {
            this.name = name;
            this.dtype = dtype;
            this.kind = kind;
            this.columnId = columnId;
            this.elementKind = elementKind;
            this.elementColumnId = elementColumnId;
            this.elementDtype = elementDtype;
        }

        boolean isList() {
            return kind == OrcProto.Type.Kind.LIST;
        }
    }

    static final class Schema {
        final List<OrcProto.Type> types; // full type tree, index = column id
        final List<Field> fields;        // top-level fields only

        Schema(List<OrcProto.Type> types, List<Field> fields) {
            this.types = types;
            this.fields = fields;
        }

        int columnCount() { return fields.size(); }
    }

    static Schema fromDataFrame(DataFrame df) {
        if (df == null || df.columnCount() == 0) {
            throw new IllegalArgumentException("DataFrame with at least one column required");
        }
        List<OrcProto.Type> types = new ArrayList<>();
        List<Field> fields = new ArrayList<>();
        OrcProto.Type.Builder root = OrcProto.Type.newBuilder().setKind(OrcProto.Type.Kind.STRUCT);
        types.add(null); // root placeholder

        for (int i = 0; i < df.columnCount(); i++) {
            Column col = df.column(i);
            Column.DType dt = col.dtype();
            String name = col.name() == null ? ("c" + i) : col.name();

            if (isListLike(dt)) {
                Column.DType elemDt = inferElementDtype(col, dt);
                OrcProto.Type.Kind elemKind = scalarDtypeToKind(elemDt);
                // element type node first
                int elemId = types.size();
                types.add(OrcProto.Type.newBuilder().setKind(elemKind).build());
                // list node
                int listId = types.size();
                types.add(OrcProto.Type.newBuilder()
                    .setKind(OrcProto.Type.Kind.LIST)
                    .addSubtypes(elemId)
                    .build());
                root.addSubtypes(listId);
                root.addFieldNames(name);
                fields.add(new Field(name, dt, OrcProto.Type.Kind.LIST, listId,
                    elemKind, elemId, elemDt));
            } else {
                OrcProto.Type.Kind kind = scalarDtypeToKind(dt);
                int colId = types.size();
                types.add(OrcProto.Type.newBuilder().setKind(kind).build());
                root.addSubtypes(colId);
                root.addFieldNames(name);
                fields.add(new Field(name, dt, kind, colId, null, -1, null));
            }
        }
        types.set(0, root.build());
        return new Schema(types, fields);
    }

    static Schema fromFooter(OrcProto.Footer footer) {
        List<OrcProto.Type> types = footer.getTypesList();
        if (types.isEmpty()) {
            throw new IllegalArgumentException("ORC footer has no types");
        }
        OrcProto.Type root = types.get(0);
        if (root.getKind() != OrcProto.Type.Kind.STRUCT) {
            throw new IllegalArgumentException("ORC root type must be STRUCT, got " + root.getKind());
        }
        List<Field> fields = new ArrayList<>();
        for (int i = 0; i < root.getSubtypesCount(); i++) {
            int colId = root.getSubtypes(i);
            if (colId < 0 || colId >= types.size()) {
                throw new IllegalArgumentException("Bad subtype id " + colId);
            }
            OrcProto.Type t = types.get(colId);
            String name = i < root.getFieldNamesCount() ? root.getFieldNames(i) : ("c" + i);
            if (t.getKind() == OrcProto.Type.Kind.LIST) {
                if (t.getSubtypesCount() < 1) {
                    throw new IllegalArgumentException("LIST type missing element subtype: " + name);
                }
                int elemId = t.getSubtypes(0);
                OrcProto.Type.Kind elemKind = types.get(elemId).getKind();
                Column.DType elemDt = kindToScalarDtype(elemKind);
                // float/double element → VECTOR; otherwise LIST
                Column.DType dt = (elemDt == Column.DType.FLOAT32 || elemDt == Column.DType.FLOAT64)
                    ? Column.DType.VECTOR : Column.DType.LIST;
                fields.add(new Field(name, dt, OrcProto.Type.Kind.LIST, colId,
                    elemKind, elemId, elemDt));
            } else if (t.getKind() == OrcProto.Type.Kind.MAP
                || t.getKind() == OrcProto.Type.Kind.STRUCT
                || t.getKind() == OrcProto.Type.Kind.UNION) {
                throw new UnsupportedOperationException(
                    "ORC nested type not supported: " + t.getKind() + " field=" + name);
            } else {
                Column.DType dt = kindToScalarDtype(t.getKind());
                fields.add(new Field(name, dt, t.getKind(), colId, null, -1, null));
            }
        }
        return new Schema(new ArrayList<>(types), fields);
    }

    static boolean isListLike(Column.DType dt) {
        return dt == Column.DType.LIST
            || dt == Column.DType.VECTOR
            || dt == Column.DType.EMBEDDING;
    }

    /** Scalar mapping only (throws on list-like / unsupported nested). */
    static OrcProto.Type.Kind scalarDtypeToKind(Column.DType dt) {
        if (dt == null) return OrcProto.Type.Kind.STRING;
        switch (dt) {
            case BOOLEAN: return OrcProto.Type.Kind.BOOLEAN;
            case INT32: return OrcProto.Type.Kind.INT;
            case INT64: return OrcProto.Type.Kind.LONG;
            case FLOAT32: return OrcProto.Type.Kind.FLOAT;
            case FLOAT64: return OrcProto.Type.Kind.DOUBLE;
            case BINARY: return OrcProto.Type.Kind.BINARY;
            case DATE: return OrcProto.Type.Kind.DATE;
            case DATETIME: return OrcProto.Type.Kind.TIMESTAMP;
            case STRING:
            case JSON:
            case TIME:
            case DURATION:
            case MAP:
            case STRUCT:
            case TENSOR:
            case GRAPH:
            case POINT_CLOUD:
            case IMAGE:
            case AUDIO:
            case VIDEO:
                // Complex non-list types stored as JSON text (STRING) for portability.
                return OrcProto.Type.Kind.STRING;
            default:
                throw new UnsupportedOperationException(
                    "ORC format path does not support scalar dtype " + dt);
        }
    }

    static Column.DType kindToScalarDtype(OrcProto.Type.Kind kind) {
        if (kind == null) return Column.DType.STRING;
        switch (kind) {
            case BOOLEAN: return Column.DType.BOOLEAN;
            case BYTE:
            case SHORT:
            case INT: return Column.DType.INT32;
            case LONG: return Column.DType.INT64;
            case FLOAT: return Column.DType.FLOAT32;
            case DOUBLE: return Column.DType.FLOAT64;
            case BINARY: return Column.DType.BINARY;
            case DATE: return Column.DType.DATE;
            case TIMESTAMP:
            case TIMESTAMP_INSTANT: return Column.DType.DATETIME;
            case STRING:
            case VARCHAR:
            case CHAR:
            default:
                return Column.DType.STRING;
        }
    }

    /** Infer list element dtype from column samples (or VECTOR/EMBEDDING defaults). */
    static Column.DType inferElementDtype(Column col, Column.DType listDtype) {
        if (listDtype == Column.DType.VECTOR || listDtype == Column.DType.EMBEDDING) {
            // Prefer float32 for ANN vectors unless first non-null is double[]
            Object sample = firstNonNull(col);
            if (sample instanceof double[]) return Column.DType.FLOAT64;
            if (sample instanceof float[]) return Column.DType.FLOAT32;
            if (sample instanceof long[]) return Column.DType.INT64;
            if (sample instanceof int[]) return Column.DType.INT32;
            // default embedding storage
            return listDtype == Column.DType.EMBEDDING ? Column.DType.FLOAT32 : Column.DType.FLOAT64;
        }
        Object sample = firstNonNull(col);
        if (sample == null) return Column.DType.INT64; // item sequences default
        if (sample instanceof long[]) return Column.DType.INT64;
        if (sample instanceof int[]) return Column.DType.INT32;
        if (sample instanceof float[]) return Column.DType.FLOAT32;
        if (sample instanceof double[]) return Column.DType.FLOAT64;
        if (sample instanceof boolean[]) return Column.DType.BOOLEAN;
        if (sample instanceof List) {
            List<?> list = (List<?>) sample;
            if (list.isEmpty()) return Column.DType.INT64;
            Object e = null;
            for (Object o : list) { if (o != null) { e = o; break; } }
            if (e instanceof Integer || e instanceof Short || e instanceof Byte) return Column.DType.INT32;
            if (e instanceof Long) return Column.DType.INT64;
            if (e instanceof Float) return Column.DType.FLOAT32;
            if (e instanceof Double) return Column.DType.FLOAT64;
            if (e instanceof Boolean) return Column.DType.BOOLEAN;
            if (e instanceof String) return Column.DType.STRING;
            if (e instanceof byte[]) return Column.DType.BINARY;
        }
        if (sample instanceof String[]) return Column.DType.STRING;
        return Column.DType.INT64;
    }

    private static Object firstNonNull(Column col) {
        if (col == null) return null;
        int n = col.size();
        for (int i = 0; i < n; i++) {
            Object v = col.get(i);
            if (v != null) return v;
        }
        return null;
    }

    // ---- list cell helpers ----

    /** Flatten a list-like cell into Object[] of boxed scalars (nulls preserved). */
    static Object[] flattenListCell(Object cell) {
        if (cell == null) return null;
        if (cell instanceof long[]) {
            long[] a = (long[]) cell;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (cell instanceof int[]) {
            int[] a = (int[]) cell;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (cell instanceof float[]) {
            float[] a = (float[]) cell;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (cell instanceof double[]) {
            double[] a = (double[]) cell;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (cell instanceof boolean[]) {
            boolean[] a = (boolean[]) cell;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (cell instanceof Object[]) {
            return (Object[]) cell;
        }
        if (cell instanceof List) {
            List<?> list = (List<?>) cell;
            return list.toArray(new Object[0]);
        }
        // single scalar → length-1 list
        return new Object[]{cell};
    }

    /** Densify element list into primitive array matching preferred dtype. */
    static Object densify(List<Object> elems, Column.DType preferred, Column.DType listDtype) {
        if (elems == null) return null;
        Column.DType pref = preferred;
        if (pref == null) {
            pref = (listDtype == Column.DType.VECTOR || listDtype == Column.DType.EMBEDDING)
                ? Column.DType.FLOAT64 : Column.DType.INT64;
        }
        int n = elems.size();
        switch (pref) {
            case INT32: {
                int[] a = new int[n];
                for (int i = 0; i < n; i++) {
                    Object e = elems.get(i);
                    a[i] = e == null ? 0 : ((Number) e).intValue();
                }
                return a;
            }
            case INT64: {
                long[] a = new long[n];
                for (int i = 0; i < n; i++) {
                    Object e = elems.get(i);
                    a[i] = e == null ? 0L : ((Number) e).longValue();
                }
                return a;
            }
            case FLOAT32: {
                float[] a = new float[n];
                for (int i = 0; i < n; i++) {
                    Object e = elems.get(i);
                    a[i] = e == null ? 0f : ((Number) e).floatValue();
                }
                return a;
            }
            case FLOAT64: {
                double[] a = new double[n];
                for (int i = 0; i < n; i++) {
                    Object e = elems.get(i);
                    a[i] = e == null ? 0d : ((Number) e).doubleValue();
                }
                return a;
            }
            case BOOLEAN: {
                boolean[] a = new boolean[n];
                for (int i = 0; i < n; i++) {
                    Object e = elems.get(i);
                    a[i] = e != null && (e instanceof Boolean ? (Boolean) e
                        : Boolean.parseBoolean(String.valueOf(e)));
                }
                return a;
            }
            case STRING: {
                String[] a = new String[n];
                for (int i = 0; i < n; i++) {
                    Object e = elems.get(i);
                    a[i] = e == null ? null : String.valueOf(e);
                }
                return a;
            }
            default:
                return new ArrayList<>(elems);
        }
    }
}
