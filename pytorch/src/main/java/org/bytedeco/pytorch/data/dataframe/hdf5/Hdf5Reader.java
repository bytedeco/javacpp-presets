package org.bytedeco.pytorch.data.dataframe.hdf5;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.hdf5.internal.Hdf5ReaderCore;
import org.bytedeco.pytorch.data.dataframe.hdf5.internal.Hdf5WriterCore;
import org.bytedeco.pytorch.data.dataframe.io.IoTypeCoercion;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * HDF5-family reader for the minimal pure-Java layout written by {@link Hdf5Writer}.
 *
 * <ul>
 *   <li>Columnar layout (group + 1-D datasets per column)</li>
 *   <li>Plain 1-D / 2-D numeric datasets</li>
 *   <li>Matrix layout ({@code values} 2-D dataset)</li>
 * </ul>
 *
 * <p>Does not depend on jhdf. Chunked/compressed third-party files are not supported.
 */
public final class Hdf5Reader {
    private Hdf5Reader() {}

    public static DataFrame read(String path, String key) throws Exception {
        return read(path, key, Hdf5Options.defaults());
    }

    public static DataFrame read(String path, String key, Hdf5Options options) throws Exception {
        Hdf5Options opt = options == null ? Hdf5Options.defaults() : options;
        Hdf5ReaderCore.Node root = Hdf5ReaderCore.open(Path.of(path));
        Hdf5ReaderCore.Node node = Hdf5ReaderCore.resolve(root, key);
        if (node == null) {
            throw new IllegalArgumentException("HDF5 key not found: " + key);
        }
        if (!node.group) {
            return fromDataset(node, leafName(key), opt);
        }
        return fromGroup(node, opt);
    }

    private static DataFrame fromGroup(Hdf5ReaderCore.Node group, Hdf5Options opt) {
        List<String> ordered = null;
        Object colNamesAttr = group.attrs.get("column_names");
        if (colNamesAttr instanceof String[]) {
            ordered = Arrays.asList((String[]) colNamesAttr);
        }

        List<String> names = new ArrayList<>();
        if (ordered != null) {
            for (String n : ordered) {
                if (group.children.containsKey(n) || group.children.containsKey(Hdf5WriterCore.sanitize(n))) {
                    String key = group.children.containsKey(n) ? n : Hdf5WriterCore.sanitize(n);
                    if (!names.contains(key)) names.add(key);
                }
            }
            for (String n : group.children.keySet()) {
                if (!names.contains(n) && !group.children.get(n).group) names.add(n);
            }
        } else {
            for (Map.Entry<String, Hdf5ReaderCore.Node> e : group.children.entrySet()) {
                if (!e.getValue().group) names.add(e.getKey());
            }
            Collections.sort(names);
        }

        if (opt.columns() != null && !opt.columns().isEmpty()) {
            names.retainAll(opt.columns());
        }

        if (names.isEmpty()) {
            if (group.children.containsKey("values") && !group.children.get("values").group) {
                return fromDataset(group.children.get("values"), "values", opt);
            }
            return DataFrame.create();
        }

        Map<String, Object[]> colData = new LinkedHashMap<>();
        Map<String, Column.DType> dtypes = new LinkedHashMap<>();
        int rowCount = -1;
        for (String name : names) {
            Hdf5ReaderCore.Node child = group.children.get(name);
            if (child == null || child.group) continue;
            Object decoded = Hdf5ReaderCore.decodeToJava(child.dataset);
            Object[] values = Hdf5ReaderCore.toObjectArray(decoded);
            if (opt.maxRows() >= 0 && values.length > opt.maxRows()) {
                values = Arrays.copyOf(values, opt.maxRows());
            }
            if (rowCount < 0) rowCount = values.length;
            else if (values.length != rowCount) {
                values = Arrays.copyOf(values, rowCount);
            }
            Column.DType dt = inferDType(name, child.dataset, values, opt, group.attrs);
            colData.put(name, values);
            dtypes.put(name, dt);
        }

        if (rowCount < 0) return DataFrame.create();
        DataFrame df = DataFrame.create();
        for (String name : names) {
            if (dtypes.containsKey(name)) df.addColumn(name, dtypes.get(name));
        }
        for (int r = 0; r < rowCount; r++) {
            int ri = df.addEmptyRow();
            for (String name : names) {
                if (!colData.containsKey(name)) continue;
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

    private static DataFrame fromDataset(Hdf5ReaderCore.Node node, String name, Hdf5Options opt) {
        Hdf5WriterCore.EncodedData ds = node.dataset;
        Object decoded = Hdf5ReaderCore.decodeToJava(ds);
        if (ds.rank == 2 && decoded instanceof double[][]) {
            double[][] m = (double[][]) decoded;
            int rows = m.length;
            int cols = rows == 0 ? 0 : m[0].length;
            if (opt.maxRows() >= 0) rows = Math.min(rows, opt.maxRows());
            DataFrame df = DataFrame.create();
            for (int c = 0; c < cols; c++) df.addColumn("col_" + c, Column.DType.FLOAT64);
            for (int r = 0; r < rows; r++) {
                int ri = df.addEmptyRow();
                for (int c = 0; c < cols; c++) df.set(ri, "col_" + c, m[r][c]);
            }
            return df;
        }
        Object[] values = Hdf5ReaderCore.toObjectArray(decoded);
        if (opt.maxRows() >= 0 && values.length > opt.maxRows()) {
            values = Arrays.copyOf(values, opt.maxRows());
        }
        Column.DType dt = inferDType(name, ds, values, opt, Map.of());
        DataFrame df = DataFrame.create();
        df.addColumn(name, dt);
        for (Object v : values) {
            int ri = df.addEmptyRow();
            df.set(ri, name, v);
        }
        return df;
    }

    private static Column.DType inferDType(String name, Hdf5WriterCore.EncodedData ds,
                                           Object[] values, Hdf5Options opt,
                                           Map<String, Object> attrs) {
        if (opt.schema() != null && opt.schema().containsKey(name)) return opt.schema().get(name);
        Object dtypesAttr = attrs.get("dtypes");
        Object namesAttr = attrs.get("column_names");
        if (dtypesAttr instanceof String[] && namesAttr instanceof String[]) {
            String[] dn = (String[]) dtypesAttr;
            String[] cn = (String[]) namesAttr;
            for (int i = 0; i < cn.length && i < dn.length; i++) {
                if (name.equals(cn[i]) || Hdf5WriterCore.sanitize(cn[i]).equals(name)) {
                    try { return Column.DType.valueOf(dn[i]); } catch (Exception ignored) {}
                }
            }
        }
        switch (ds.dtypeCode) {
            case 1: return Column.DType.INT32;
            case 2: return Column.DType.INT64;
            case 3: return Column.DType.FLOAT32;
            case 4: return Column.DType.FLOAT64;
            case 5: return Column.DType.BOOLEAN;
            case 6: return Column.DType.STRING;
            default: break;
        }
        Column.DType acc = null;
        int sample = Math.min(1000, values.length);
        for (int i = 0; i < sample; i++) {
            if (values[i] == null) continue;
            Column.DType t = IoTypeCoercion.inferFromObject(values[i]);
            acc = acc == null ? t : IoTypeCoercion.widen(acc, t);
        }
        return acc == null ? Column.DType.STRING : acc;
    }

    private static String leafName(String key) {
        if (key == null || key.isEmpty() || "/".equals(key)) return "value";
        int slash = key.lastIndexOf('/');
        String leaf = slash >= 0 ? key.substring(slash + 1) : key;
        return leaf.isEmpty() ? "value" : leaf;
    }
}
