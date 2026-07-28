package org.bytedeco.pytorch.dataframe;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/** Default in-heap column storage. */
final class ListStorage implements ColumnStorage {
    private final Column.DType dtype;
    private final List<Object> data;

    ListStorage(Column.DType dtype) {
        this.dtype = dtype;
        this.data = new ArrayList<>();
    }

    ListStorage(Column.DType dtype, List<Object> data) {
        this.dtype = dtype;
        this.data = new ArrayList<>(data);
    }

    @Override public int size() { return data.size(); }

    @Override public Object get(int index) {
        if (index < 0) index = data.size() + index;
        return data.get(index);
    }

    @Override public void set(int index, Object value) {
        if (index < 0) index = data.size() + index;
        data.set(index, value);
    }

    @Override public void add(Object value) { data.add(value); }

    @Override public void addAll(Collection<?> values) { data.addAll(values); }

    @Override public Column.DType dtype() { return dtype; }

    @Override public ColumnStorage copy() {
        return new ListStorage(dtype, new ArrayList<>(data));
    }

    @Override public List<Object> materialize() { return data; }
}
