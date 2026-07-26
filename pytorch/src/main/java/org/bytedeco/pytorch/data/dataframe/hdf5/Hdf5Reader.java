package org.bytedeco.pytorch.data.dataframe.hdf5;

import io.jhdf.HdfFile;
import io.jhdf.api.Attribute;
import io.jhdf.api.Dataset;
import io.jhdf.api.Group;
import io.jhdf.api.Node;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.io.IoTypeCoercion;

import java.nio.file.Path;
import java.util.*;

/**
 * HDF5 reader supporting:
 * <ul>
 *   <li>Our columnar layout (group + 1-D datasets per column)</li>
 *   <li>Plain 1-D / 2-D numeric datasets</li>
 *   <li>Best-effort pandas fixed-style groups when recognizable</li>
 * </ul>
 */
public final class Hdf5Reader {
    private Hdf5Reader() {}

    public static DataFrame read(String path, String key) throws Exception {
        return read(path, key, Hdf5Options.defaults());
    }

    public static DataFrame read(String path, String key, Hdf5Options options) throws Exception {
        Hdf5Options opt = options == null ? Hdf5Options.defaults() : options;
        String k = normalizeKey(key);
        try (HdfFile file = new HdfFile(Path.of(path))) {
            Node node = resolve(file, k);
            if (node == null) {
                throw new IllegalArgumentException("HDF5 key not found: " + key);
            }
            if (node instanceof Dataset) {
                return fromDataset((Dataset) node, leafName(k), opt);
            }
            if (node instanceof Group) {
                return fromGroup((Group) node, opt);
            }
            throw new IllegalArgumentException("Unsupported HDF5 node type at " + key + ": " + node.getType());
        }
    }

    private static Node resolve(HdfFile file, String key) {
        if (key == null || key.isEmpty() || "/".equals(key)) return file;
        String path = key.startsWith("/") ? key.substring(1) : key;
        if (path.isEmpty()) return file;
        try {
            return file.getByPath(path);
        } catch (Exception e) {
            // try dataset path API
            try {
                return file.getDatasetByPath(path);
            } catch (Exception e2) {
                return null;
            }
        }
    }

    private static DataFrame fromGroup(Group group, Hdf5Options opt) {
        // Detect format attribute
        String format = attrString(group, "format");
        if (format == null) format = attrString(group, "pandas_type");

        // Column order from attribute if present
        List<String> ordered = null;
        Object colNamesAttr = attrData(group, "column_names");
        if (colNamesAttr instanceof String[]) {
            ordered = Arrays.asList((String[]) colNamesAttr);
        } else if (colNamesAttr instanceof Object[]) {
            ordered = new ArrayList<>();
            for (Object o : (Object[]) colNamesAttr) ordered.add(String.valueOf(o));
        }

        Map<String, Node> children = group.getChildren();
        List<String> names = new ArrayList<>();
        if (ordered != null) {
            for (String n : ordered) if (children.containsKey(n)) names.add(n);
            for (String n : children.keySet()) if (!names.contains(n) && children.get(n) instanceof Dataset) names.add(n);
        } else {
            for (Map.Entry<String, Node> e : children.entrySet()) {
                if (e.getValue() instanceof Dataset) names.add(e.getKey());
            }
            // stable-ish order
            Collections.sort(names);
        }

        if (opt.columns() != null && !opt.columns().isEmpty()) {
            names.retainAll(opt.columns());
        }

        if (names.isEmpty()) {
            // maybe nested values dataset (pandas-ish)
            if (children.containsKey("values") && children.get("values") instanceof Dataset) {
                return fromDataset((Dataset) children.get("values"), "values", opt);
            }
            return DataFrame.create();
        }

        // Read each 1-D dataset as a column
        Map<String, Object[]> colData = new LinkedHashMap<>();
        Map<String, Column.DType> dtypes = new LinkedHashMap<>();
        int rowCount = -1;
        for (String name : names) {
            Dataset ds = (Dataset) children.get(name);
            Object[] values = flattenDataset(ds);
            if (opt.maxRows() >= 0 && values.length > opt.maxRows()) {
                values = Arrays.copyOf(values, opt.maxRows());
            }
            if (rowCount < 0) rowCount = values.length;
            else if (values.length != rowCount) {
                // pad / truncate
                values = Arrays.copyOf(values, rowCount);
            }
            Column.DType dt = inferDType(ds, values, opt, name);
            colData.put(name, values);
            dtypes.put(name, dt);
        }

        DataFrame df = DataFrame.create();
        for (String name : names) df.addColumn(name, dtypes.get(name));
        for (int r = 0; r < rowCount; r++) {
            int ri = df.addEmptyRow();
            for (String name : names) {
                Object v = colData.get(name)[r];
                try {
                    df.set(ri, name, v == null ? null : IoTypeCoercion.coerce(v, dtypes.get(name)));
                } catch (Exception ex) {
                    df.set(ri, name, v == null ? null : String.valueOf(v));
                }
            }
        }
        return df;
    }

    private static DataFrame fromDataset(Dataset ds, String name, Hdf5Options opt) {
        int[] dims = ds.getDimensions();
        if (dims == null || dims.length == 0) {
            // scalar
            DataFrame df = DataFrame.create();
            Object data = ds.getData();
            Column.DType dt = javaToDType(ds.getJavaType(), data);
            df.addColumn(name, dt);
            int ri = df.addEmptyRow();
            df.set(ri, name, boxScalar(data));
            return df;
        }
        if (dims.length == 1) {
            Object[] values = flattenDataset(ds);
            if (opt.maxRows() >= 0 && values.length > opt.maxRows()) {
                values = Arrays.copyOf(values, opt.maxRows());
            }
            Column.DType dt = inferDType(ds, values, opt, name);
            DataFrame df = DataFrame.create();
            df.addColumn(name, dt);
            for (Object v : values) {
                int ri = df.addEmptyRow();
                df.set(ri, name, v);
            }
            return df;
        }
        if (dims.length == 2) {
            // rows x cols matrix
            Object raw = ds.getData();
            int rows = dims[0];
            int cols = dims[1];
            if (opt.maxRows() >= 0) rows = Math.min(rows, opt.maxRows());
            DataFrame df = DataFrame.create();
            Column.DType dt = javaToDType(ds.getJavaType(), null);
            for (int c = 0; c < cols; c++) df.addColumn("col_" + c, dt);
            // raw is typically multi-dim array
            for (int r = 0; r < rows; r++) {
                int ri = df.addEmptyRow();
                for (int c = 0; c < cols; c++) {
                    Object v = matrixGet(raw, r, c, dims);
                    df.set(ri, "col_" + c, v);
                }
            }
            return df;
        }
        // higher rank → flatten
        Object[] values = flattenDataset(ds);
        Column.DType dt = inferDType(ds, values, opt, name);
        DataFrame df = DataFrame.create();
        df.addColumn(name, dt);
        for (Object v : values) {
            int ri = df.addEmptyRow();
            df.set(ri, name, v);
        }
        return df;
    }

    private static Object[] flattenDataset(Dataset ds) {
        Object flat = null;
        try {
            flat = ds.getDataFlat();
        } catch (Exception ignored) {}
        if (flat == null) flat = ds.getData();
        return toObjectArray(flat);
    }

    private static Object[] toObjectArray(Object data) {
        if (data == null) return new Object[0];
        if (data instanceof Object[]) {
            Object[] arr = (Object[]) data;
            // nested?
            if (arr.length > 0 && arr[0] != null && arr[0].getClass().isArray()) {
                List<Object> out = new ArrayList<>();
                flattenInto(arr, out);
                return out.toArray();
            }
            return arr;
        }
        if (data instanceof double[]) {
            double[] a = (double[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (data instanceof float[]) {
            float[] a = (float[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (data instanceof long[]) {
            long[] a = (long[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (data instanceof int[]) {
            int[] a = (int[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = (long) a[i];
            return o;
        }
        if (data instanceof short[]) {
            short[] a = (short[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = (long) a[i];
            return o;
        }
        if (data instanceof byte[]) {
            byte[] a = (byte[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = (long) a[i];
            return o;
        }
        if (data instanceof boolean[]) {
            boolean[] a = (boolean[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (data instanceof String[]) return (String[]) data;
        if (data.getClass().isArray()) {
            int n = java.lang.reflect.Array.getLength(data);
            Object[] o = new Object[n];
            for (int i = 0; i < n; i++) o[i] = java.lang.reflect.Array.get(data, i);
            return o;
        }
        return new Object[]{data};
    }

    private static void flattenInto(Object arr, List<Object> out) {
        if (arr == null) { out.add(null); return; }
        if (!arr.getClass().isArray()) { out.add(arr); return; }
        if (arr instanceof Object[]) {
            for (Object o : (Object[]) arr) flattenInto(o, out);
            return;
        }
        int n = java.lang.reflect.Array.getLength(arr);
        for (int i = 0; i < n; i++) out.add(java.lang.reflect.Array.get(arr, i));
    }

    private static Object matrixGet(Object raw, int r, int c, int[] dims) {
        if (raw == null) return null;
        if (raw instanceof Object[]) {
            Object row = ((Object[]) raw)[r];
            if (row == null) return null;
            if (row.getClass().isArray()) return java.lang.reflect.Array.get(row, c);
            return row;
        }
        // flat primitive
        int idx = r * dims[1] + c;
        Object[] flat = toObjectArray(raw);
        return idx < flat.length ? flat[idx] : null;
    }

    private static Column.DType inferDType(Dataset ds, Object[] values, Hdf5Options opt, String name) {
        if (opt.schema() != null && opt.schema().containsKey(name)) return opt.schema().get(name);
        Column.DType fromJava = javaToDType(ds.getJavaType(), values.length > 0 ? values[0] : null);
        if (fromJava != Column.DType.STRING) return fromJava;
        Column.DType acc = null;
        int sample = Math.min(1000, values.length);
        for (int i = 0; i < sample; i++) {
            if (values[i] == null) continue;
            Column.DType t = IoTypeCoercion.inferFromObject(values[i]);
            acc = acc == null ? t : IoTypeCoercion.widen(acc, t);
        }
        return acc == null ? Column.DType.STRING : acc;
    }

    private static Column.DType javaToDType(Class<?> cls, Object sample) {
        if (cls == null && sample != null) cls = sample.getClass();
        if (cls == null) return Column.DType.STRING;
        if (cls == boolean.class || cls == Boolean.class || cls == boolean[].class) return Column.DType.BOOLEAN;
        if (cls == int.class || cls == Integer.class || cls == int[].class) return Column.DType.INT32;
        if (cls == long.class || cls == Long.class || cls == long[].class) return Column.DType.INT64;
        if (cls == short.class || cls == Short.class || cls == byte.class || cls == Byte.class
            || cls == short[].class || cls == byte[].class) return Column.DType.INT64;
        if (cls == float.class || cls == Float.class || cls == float[].class) return Column.DType.FLOAT32;
        if (cls == double.class || cls == Double.class || cls == double[].class) return Column.DType.FLOAT64;
        if (cls == String.class || cls == String[].class) return Column.DType.STRING;
        return Column.DType.STRING;
    }

    private static Object boxScalar(Object data) {
        if (data == null) return null;
        if (data.getClass().isArray() && java.lang.reflect.Array.getLength(data) == 1) {
            return java.lang.reflect.Array.get(data, 0);
        }
        return data;
    }

    private static String attrString(Node node, String name) {
        Object d = attrData(node, name);
        return d == null ? null : String.valueOf(d instanceof Object[] ? ((Object[]) d)[0] : d);
    }

    private static Object attrData(Node node, String name) {
        try {
            Attribute a = node.getAttribute(name);
            return a == null ? null : a.getData();
        } catch (Exception e) {
            return null;
        }
    }

    private static String normalizeKey(String key) {
        if (key == null || key.isEmpty()) return "/";
        if (!key.startsWith("/")) return "/" + key;
        return key;
    }

    private static String leafName(String key) {
        String k = normalizeKey(key);
        int idx = k.lastIndexOf('/');
        String leaf = idx >= 0 ? k.substring(idx + 1) : k;
        return leaf.isEmpty() ? "data" : leaf;
    }
}
