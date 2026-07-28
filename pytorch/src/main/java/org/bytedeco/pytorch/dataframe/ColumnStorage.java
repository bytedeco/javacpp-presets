package org.bytedeco.pytorch.dataframe;

import java.util.Collection;
import java.util.List;

/**
 * Package-private column storage backend.
 * {@link ListStorage} is the default; {@link ArrowStorage} enables zero-copy IPC.
 */
public interface ColumnStorage {
    int size();
    Object get(int index);
    void set(int index, Object value);
    void add(Object value);
    void addAll(Collection<?> values);
    Column.DType dtype();
    ColumnStorage copy();
    /** Materialize to a mutable list view (may copy). */
    List<Object> materialize();
    /** True if backed by Arrow FieldVector. */
    default boolean isArrowBacked() { return false; }
    /** Arrow vector if present; caller must not free unless ownership transferred. */
    default org.apache.arrow.vector.FieldVector arrowVectorOrNull() { return null; }
    /** Release native resources if any. */
    default void close() {}
}
