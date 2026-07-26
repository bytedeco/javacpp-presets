package org.bytedeco.pytorch.data.dataframe;

import java.util.*;

/**
 * A single column in a DataFrame.
 * Supports primitive types, strings, temporal types, torch Tensor values,
 * and optional Arrow-backed zero-copy storage.
 */
public final class Column implements AutoCloseable {
    /** Column data type. */
    public enum DType {
        INT32, INT64, FLOAT32, FLOAT64, BOOLEAN, STRING, TENSOR,
        DATE, DATETIME, TIME, DURATION,
        /** Dense float vector (cell value is float[]). Used for ANN/HNSW. */
        VECTOR,
        /** Multimodal / structured cell wrappers ({@code DataValue}). */
        IMAGE, AUDIO, VIDEO, EMBEDDING, BINARY, JSON,
        LIST, MAP, STRUCT, GRAPH, POINT_CLOUD
    }

    private final String name;
    private ColumnStorage storage;

    public Column(String name, DType dtype) {
        this.name = name;
        this.storage = new ListStorage(dtype);
    }

    public Column(String name, DType dtype, List<Object> data) {
        this.name = name;
        this.storage = new ListStorage(dtype, data);
    }

    /** Construct from an existing storage backend (Arrow or list). */
    public Column(String name, ColumnStorage storage) {
        this.name = name;
        this.storage = storage;
    }

    public String name() { return name; }
    public DType dtype() { return storage.dtype(); }
    public int size() { return storage.size(); }
    public boolean isArrowBacked() { return storage.isArrowBacked(); }

    /** Mutable list view — materializes Arrow-backed columns. */
    public List<Object> data() { return storage.materialize(); }

    public Object get(int index) { return storage.get(index); }

    public void set(int index, Object value) { storage.set(index, value); }

    public void add(Object value) { storage.add(value); }

    public void addAll(Collection<?> values) { storage.addAll(values); }

    ColumnStorage storage() { return storage; }

    /** Returns a view of the data as doubles (numeric columns only). NaN for non-numeric. */
    public List<Double> asDoubleList() {
        int n = size();
        List<Double> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Object v = get(i);
            if (v == null) out.add(Double.NaN);
            else if (v instanceof Number) out.add(((Number) v).doubleValue());
            else out.add(Double.NaN);
        }
        return out;
    }

    /** Returns a view of the data as a double array (numeric columns only). */
    public double[] asDoubleArray() {
        int n = size();
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            Object v = get(i);
            if (v instanceof Number) out[i] = ((Number) v).doubleValue();
            else out[i] = Double.NaN;
        }
        return out;
    }

    /** Deep-copy the column (Arrow → list copy). */
    public Column copy() {
        return new Column(name, storage.copy());
    }

    /** Rename without copying data (shares storage — prefer copy+rename for safety). */
    public Column rename(String newName) {
        return new Column(newName, storage);
    }

    @Override
    public void close() {
        storage.close();
    }

    @Override
    public String toString() {
        int n = size();
        if (n == 0) return name + ":[]";
        int show = Math.min(5, n);
        StringBuilder sb = new StringBuilder(name).append(":").append(dtype()).append("[");
        for (int i = 0; i < show; i++) {
            if (i > 0) sb.append(", ");
            Object v = get(i);
            sb.append(v == null ? "null" : v.toString());
        }
        if (n > show) sb.append(", ... (").append(n).append(" total)");
        sb.append("]");
        if (isArrowBacked()) sb.append("{arrow}");
        return sb.toString();
    }
}
