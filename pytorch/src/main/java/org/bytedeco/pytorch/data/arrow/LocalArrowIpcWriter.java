package org.bytedeco.pytorch.data.arrow;

import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
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
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowFileWriter;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

/**
 * Local-only Arrow IPC / Feather v2 writer.
 */
public final class LocalArrowIpcWriter {

    private LocalArrowIpcWriter() {}

    public static void write(DataFrame df, String path) throws Exception {
        List<Field> fields = new ArrayList<>();
        for (Column c : df.columns()) {
            fields.add(ArrowSchemaMapper.toField(c.name(), c.dtype()));
        }
        Schema schema = new Schema(fields);

        try (BufferAllocator allocator = new RootAllocator();
             VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
             FileChannel channel = FileChannel.open(Path.of(path),
                     StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
             ArrowFileWriter writer = new ArrowFileWriter(root, null, channel)) {

            writer.start();
            int n = df.rowCount();
            root.setRowCount(n);

            List<FieldVector> vectors = root.getFieldVectors();
            for (int ci = 0; ci < vectors.size(); ci++) {
                fillVector(vectors.get(ci), df.column(ci), n);
            }

            writer.writeBatch();
            writer.end();
        }
    }

    private static void fillVector(FieldVector vec, Column col, int n) {
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
        } else if (vec instanceof VarCharVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else v.setSafe(i, val.toString().getBytes(StandardCharsets.UTF_8));
            }
            v.setValueCount(n);
        } else {
            for (int i = 0; i < n; i++) vec.setNull(i);
            vec.setValueCount(n);
        }
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
