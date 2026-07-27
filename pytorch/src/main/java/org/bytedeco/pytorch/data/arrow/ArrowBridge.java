/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.data.arrow;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.bytedeco.pytorch.JvmModuleSupport;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Bidirectional bridge between {@link DataFrame} and Apache Arrow
 * {@link VectorSchemaRoot} / {@link ArrowReader}.
 *
 * <p>Used by official Lance ({@code org.lance}) and any consumer that speaks Arrow IPC.
 */
public final class ArrowBridge {

    private ArrowBridge() {}

    /** Convert a DataFrame into a single-batch {@link VectorSchemaRoot} (caller closes root). */
    public static VectorSchemaRoot toRoot(DataFrame df, BufferAllocator allocator) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(allocator, "allocator");
        JvmModuleSupport.ensureNioBufferAccess();
        List<Field> fields = new ArrayList<>();
        for (Column c : df.columns()) {
            fields.add(ArrowSchemaMapper.toField(c.name(), c.dtype()));
        }
        Schema schema = new Schema(fields);
        VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
        int n = df.rowCount();
        root.setRowCount(n);
        List<FieldVector> vectors = root.getFieldVectors();
        for (int ci = 0; ci < vectors.size(); ci++) {
            fillVector(vectors.get(ci), df.column(ci), n);
        }
        return root;
    }

    /**
     * One-shot ArrowReader over a DataFrame (single record batch).
     * Implementation serializes to an in-memory Arrow stream so it works with
     * any Arrow consumer (including Lance {@code Dataset.write().reader(...)}).
     */
    public static ArrowReader toArrowReader(DataFrame df, BufferAllocator allocator) throws Exception {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(allocator, "allocator");
        byte[] ipc;
        try (VectorSchemaRoot root = toRoot(df, allocator);
             ByteArrayOutputStream bos = new ByteArrayOutputStream();
             ArrowStreamWriter writer = new ArrowStreamWriter(root, null, bos)) {
            writer.start();
            writer.writeBatch();
            writer.end();
            ipc = bos.toByteArray();
        }
        return new ArrowStreamReader(new ByteArrayInputStream(ipc), allocator);
    }

    /** Materialize all batches from an {@link ArrowReader} into a DataFrame. */
    public static DataFrame fromArrowReader(ArrowReader reader) throws Exception {
        Objects.requireNonNull(reader, "reader");
        DataFrame df = null;
        while (reader.loadNextBatch()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            DataFrame batch = fromRoot(root);
            if (df == null) {
                df = batch;
            } else {
                append(df, batch);
            }
        }
        return df == null ? DataFrame.create() : df;
    }

    /** Convert a single {@link VectorSchemaRoot} batch to a DataFrame (copies values). */
    public static DataFrame fromRoot(VectorSchemaRoot root) {
        Objects.requireNonNull(root, "root");
        DataFrame df = DataFrame.create();
        List<FieldVector> vectors = root.getFieldVectors();
        int n = root.getRowCount();
        for (FieldVector vec : vectors) {
            Column.DType dt = ArrowSchemaMapper.fromField(vec.getField());
            df.addColumn(vec.getName(), dt);
        }
        for (int r = 0; r < n; r++) {
            Object[] row = new Object[vectors.size()];
            for (int c = 0; c < vectors.size(); c++) {
                row[c] = getValue(vectors.get(c), r);
            }
            df.addRow(row);
        }
        return df;
    }

    /** Convenience: DataFrame → bytes (Arrow stream IPC). */
    public static byte[] toIpcBytes(DataFrame df) throws Exception {
        try (BufferAllocator alloc = new RootAllocator();
             VectorSchemaRoot root = toRoot(df, alloc);
             ByteArrayOutputStream bos = new ByteArrayOutputStream();
             ArrowStreamWriter writer = new ArrowStreamWriter(root, null, bos)) {
            writer.start();
            writer.writeBatch();
            writer.end();
            return bos.toByteArray();
        }
    }

    /** Convenience: Arrow stream IPC bytes → DataFrame. */
    public static DataFrame fromIpcBytes(byte[] ipc) throws Exception {
        try (BufferAllocator alloc = new RootAllocator();
             ArrowStreamReader reader = new ArrowStreamReader(new ByteArrayInputStream(ipc), alloc)) {
            return fromArrowReader(reader);
        }
    }

    // ---- internals -------------------------------------------------------

    private static void append(DataFrame dst, DataFrame src) {
        List<Column> cols = dst.columns();
        for (int r = 0; r < src.rowCount(); r++) {
            Object[] row = new Object[cols.size()];
            for (int c = 0; c < cols.size(); c++) {
                String name = cols.get(c).name();
                row[c] = src.hasColumn(name) ? src.get(r, name) : null;
            }
            dst.addRow(row);
        }
    }

    private static Object getValue(FieldVector vec, int i) {
        if (vec.isNull(i)) return null;
        if (vec instanceof IntVector v) return v.get(i);
        if (vec instanceof BigIntVector v) return v.get(i);
        if (vec instanceof Float4Vector v) return v.get(i);
        if (vec instanceof Float8Vector v) return v.get(i);
        if (vec instanceof BitVector v) return v.get(i) == 1;
        if (vec instanceof VarCharVector v) {
            byte[] b = v.get(i);
            return b == null ? null : new String(b, StandardCharsets.UTF_8);
        }
        // fallback
        Object o = vec.getObject(i);
        return o == null ? null : o.toString();
    }

    static void fillVector(FieldVector vec, Column col, int n) {
        vec.setInitialCapacity(n);
        vec.allocateNew();
        if (vec instanceof IntVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else v.setSafe(i, ((Number) val).intValue());
            }
        } else if (vec instanceof BigIntVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else if (val instanceof Duration d) v.setSafe(i, d.toMillis());
                else v.setSafe(i, ((Number) val).longValue());
            }
        } else if (vec instanceof Float4Vector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else v.setSafe(i, ((Number) val).floatValue());
            }
        } else if (vec instanceof Float8Vector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else v.setSafe(i, ((Number) val).doubleValue());
            }
        } else if (vec instanceof BitVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else {
                    boolean b = val instanceof Boolean bo ? bo
                            : val instanceof Number num && num.intValue() != 0;
                    v.setSafe(i, b ? 1 : 0);
                }
            }
        } else if (vec instanceof VarCharVector v) {
            for (int i = 0; i < n; i++) {
                Object val = col.get(i);
                if (val == null) v.setNull(i);
                else {
                    byte[] bytes = String.valueOf(val).getBytes(StandardCharsets.UTF_8);
                    v.setSafe(i, bytes);
                }
            }
        } else {
            // best-effort string
            if (vec instanceof org.apache.arrow.vector.DateDayVector v) {
                for (int i = 0; i < n; i++) {
                    Object val = col.get(i);
                    if (val == null) v.setNull(i);
                    else if (val instanceof Number num) v.setSafe(i, num.intValue());
                    else if (val instanceof LocalDate ld) v.setSafe(i, (int) ld.toEpochDay());
                    else v.setNull(i);
                }
            }
        }
        vec.setValueCount(n);
    }
}
