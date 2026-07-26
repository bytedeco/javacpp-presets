package org.bytedeco.pytorch.data.dataframe.ann;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

/**
 * Helpers for VECTOR columns (cell type {@code float[]}).
 */
public final class VectorColumn {
    private VectorColumn() {}

    /** Pack a VECTOR column into a contiguous row-major float matrix [n * dim]. */
    public static float[] pack(Column col) {
        int n = col.size();
        if (n == 0) return new float[0];
        int dim = -1;
        float[][] rows = new float[n][];
        for (int i = 0; i < n; i++) {
            float[] v = asFloatArray(col.get(i));
            rows[i] = v;
            if (v != null) {
                if (dim < 0) dim = v.length;
                else if (v.length != dim) {
                    throw new IllegalArgumentException(
                        "Inconsistent vector dim at row " + i + ": " + v.length + " vs " + dim);
                }
            }
        }
        if (dim < 0) dim = 0;
        float[] matrix = new float[n * dim];
        for (int i = 0; i < n; i++) {
            float[] v = rows[i];
            if (v != null) System.arraycopy(v, 0, matrix, i * dim, dim);
            // null → zeros
        }
        return matrix;
    }

    public static int dimOf(Column col) {
        for (int i = 0; i < col.size(); i++) {
            float[] v = asFloatArray(col.get(i));
            if (v != null) return v.length;
        }
        return 0;
    }

    public static float[] asFloatArray(Object v) {
        if (v == null) return null;
        if (v instanceof float[]) return (float[]) v;
        if (v instanceof double[]) {
            double[] d = (double[]) v;
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            return f;
        }
        if (v instanceof Number[]) {
            Number[] arr = (Number[]) v;
            float[] f = new float[arr.length];
            for (int i = 0; i < arr.length; i++) f[i] = arr[i] == null ? 0f : arr[i].floatValue();
            return f;
        }
        // Multimodal cell types used by DataFrame TENSOR/VECTOR/EMBEDDING columns
        float[] viaBridge = org.bytedeco.pytorch.data.dataframe.tensor.TensorBridge.asFloatVector(v);
        if (viaBridge != null) return viaBridge;
        throw new IllegalArgumentException("Not a vector cell: " + v.getClass());
    }

    /** Build a one-column DataFrame of vectors (optional parallel id column). */
    public static DataFrame fromVectors(String vectorCol, float[][] data) {
        return fromVectors(vectorCol, data, null, null);
    }

    public static DataFrame fromVectors(String vectorCol, float[][] data, String idCol, long[] ids) {
        DataFrame df = DataFrame.create();
        if (idCol != null) {
            df.addColumn(idCol, Column.DType.INT64);
        }
        df.addColumn(vectorCol, Column.DType.VECTOR);
        for (int i = 0; i < data.length; i++) {
            if (idCol != null) {
                long id = ids != null && i < ids.length ? ids[i] : i;
                df.addRow(id, data[i] == null ? null : data[i].clone());
            } else {
                df.addRow((Object) (data[i] == null ? null : data[i].clone()));
            }
        }
        return df;
    }

    /** L2-normalize rows of a packed matrix in-place. */
    public static void l2Normalize(float[] matrix, int n, int dim) {
        for (int r = 0; r < n; r++) {
            int base = r * dim;
            float sum = 0f;
            for (int i = 0; i < dim; i++) sum += matrix[base + i] * matrix[base + i];
            if (sum <= 0f) continue;
            float inv = (float) (1.0 / Math.sqrt(sum));
            for (int i = 0; i < dim; i++) matrix[base + i] *= inv;
        }
    }
}
