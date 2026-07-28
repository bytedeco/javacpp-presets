package org.bytedeco.pytorch.dataframe;

import java.util.*;

/**
 * Ordered column schema: name → {@link Column.DType}.
 */
public final class Schema {
    private final List<String> names;
    private final List<Column.DType> types;
    private final Map<String, Integer> index;

    public Schema() {
        this.names = new ArrayList<>();
        this.types = new ArrayList<>();
        this.index = new LinkedHashMap<>();
    }

    public Schema(List<String> names, List<Column.DType> types) {
        if (names.size() != types.size())
            throw new IllegalArgumentException("names/types size mismatch");
        this.names = new ArrayList<>(names);
        this.types = new ArrayList<>(types);
        this.index = new LinkedHashMap<>();
        for (int i = 0; i < names.size(); i++) this.index.put(names.get(i), i);
    }

    public static Schema of(String name, Column.DType dtype) {
        Schema s = new Schema();
        s.add(name, dtype);
        return s;
    }

    public static Schema fromDataFrame(DataFrame df) {
        Schema s = new Schema();
        for (Column c : df.columns()) s.add(c.name(), c.dtype());
        return s;
    }

    public Schema add(String name, Column.DType dtype) {
        if (index.containsKey(name)) throw new IllegalArgumentException("duplicate field: " + name);
        index.put(name, names.size());
        names.add(name);
        types.add(dtype);
        return this;
    }

    public int size() { return names.size(); }
    public List<String> fieldNames() { return Collections.unmodifiableList(names); }
    public List<Column.DType> fieldTypes() { return Collections.unmodifiableList(types); }

    public String fieldName(int i) { return names.get(i); }
    public Column.DType fieldType(int i) { return types.get(i); }
    public Column.DType fieldType(String name) {
        Integer i = index.get(name);
        if (i == null) throw new IllegalArgumentException("no field: " + name);
        return types.get(i);
    }

    public boolean hasField(String name) { return index.containsKey(name); }
    public OptionalInt fieldIndex(String name) {
        Integer i = index.get(name);
        return i == null ? OptionalInt.empty() : OptionalInt.of(i);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Schema{");
        for (int i = 0; i < names.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(names.get(i)).append(':').append(types.get(i));
        }
        return sb.append('}').toString();
    }
}
