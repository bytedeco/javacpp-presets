package org.bytedeco.pytorch.data.dataframe;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeMilliVector;
import org.apache.arrow.vector.TimeStampMilliVector;
import org.apache.arrow.vector.VarCharVector;

/**
 * Zero-copy Arrow-backed column storage. Mutations trigger copy-on-write into {@link ListStorage}.
 */
public final class ArrowStorage implements ColumnStorage {
    private final Column.DType dtype;
    private FieldVector vector;
    private final BufferAllocator allocator;
    private final boolean ownVector;
    private boolean closed;
    private ListStorage cow;

    public ArrowStorage(Column.DType dtype, FieldVector vector, BufferAllocator allocator) {
        this(dtype, vector, allocator, true);
    }

    public ArrowStorage(Column.DType dtype, FieldVector vector, BufferAllocator allocator, boolean ownVector) {
        this.dtype = dtype;
        this.vector = vector;
        this.allocator = allocator;
        this.ownVector = ownVector;
        this.closed = false;
    }

    @Override public boolean isArrowBacked() { return cow == null && vector != null; }

    @Override public FieldVector arrowVectorOrNull() {
        return cow == null ? vector : null;
    }

    @Override public int size() {
        if (cow != null) return cow.size();
        return vector.getValueCount();
    }

    @Override public Object get(int index) {
        if (cow != null) return cow.get(index);
        if (index < 0) index = size() + index;
        if (vector.isNull(index)) return null;
        return readValue(vector, index, dtype);
    }

    @Override public void set(int index, Object value) {
        ensureMutable().set(index, value);
    }

    @Override public void add(Object value) {
        ensureMutable().add(value);
    }

    @Override public void addAll(Collection<?> values) {
        ensureMutable().addAll(values);
    }

    @Override public Column.DType dtype() { return dtype; }

    @Override public ColumnStorage copy() {
        if (cow != null) return cow.copy();
        return new ListStorage(dtype, materialize());
    }

    @Override public List<Object> materialize() {
        if (cow != null) return cow.materialize();
        int n = size();
        List<Object> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(get(i));
        return out;
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        if (cow == null && vector != null && ownVector) {
            try { vector.close(); } catch (Exception ignored) {}
            vector = null;
        }
    }

    private ListStorage ensureMutable() {
        if (cow != null) return cow;
        cow = new ListStorage(dtype, materialize());
        if (ownVector) {
            try { if (vector != null) vector.close(); } catch (Exception ignored) {}
        }
        vector = null;
        return cow;
    }

    public static Object readValue(FieldVector vec, int index, Column.DType dtype) {
        if (vec instanceof IntVector v) return v.get(index);
        if (vec instanceof BigIntVector v) return v.get(index);
        if (vec instanceof Float4Vector v) return v.get(index);
        if (vec instanceof Float8Vector v) return v.get(index);
        if (vec instanceof BitVector v) return v.get(index) == 1;
        if (vec instanceof VarCharVector v) {
            byte[] b = v.get(index);
            return b == null ? null : new String(b, StandardCharsets.UTF_8);
        }
        if (vec instanceof DateDayVector v) {
            return LocalDate.ofEpochDay(v.get(index));
        }
        if (vec instanceof TimeStampMilliVector v) {
            return Instant.ofEpochMilli(v.get(index));
        }
        if (vec instanceof TimeMilliVector v) {
            int millis = v.get(index);
            return LocalTime.ofNanoOfDay(millis * 1_000_000L);
        }
        Object o = vec.getObject(index);
        if (o == null) return null;
        return switch (dtype) {
            case DATE -> {
                if (o instanceof LocalDate ld) yield ld;
                if (o instanceof Number n) yield LocalDate.ofEpochDay(n.intValue());
                yield LocalDate.parse(o.toString());
            }
            case DATETIME -> {
                if (o instanceof Instant in) yield in;
                if (o instanceof LocalDateTime ldt) yield ldt.toInstant(ZoneOffset.UTC);
                if (o instanceof Number n) yield Instant.ofEpochMilli(n.longValue());
                yield Instant.parse(o.toString());
            }
            case TIME -> {
                if (o instanceof LocalTime lt) yield lt;
                yield LocalTime.parse(o.toString());
            }
            default -> o;
        };
    }
}
