package org.bytedeco.pytorch.data.arrow;

import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowFileWriter;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.bytedeco.pytorch.JvmModuleSupport;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

/**
 * Local-only Arrow IPC / Feather v2 writer.
 * Complex columns (LIST / VECTOR / EMBEDDING / MAP / STRUCT) use native nested Arrow types
 * via {@link ArrowSchemaMapper} + {@link ArrowComplexVectors}.
 */
public final class LocalArrowIpcWriter {

    private LocalArrowIpcWriter() {}

    public static void write(DataFrame df, String path) throws Exception {
        // Fail fast with copy-paste JVM flags if java.nio is not open (Arrow MemoryUtil).
        JvmModuleSupport.ensureNioBufferAccess();
        List<Field> fields = new ArrayList<>();
        for (Column c : df.columns()) {
            fields.add(ArrowSchemaMapper.toField(c));
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
                ArrowComplexVectors.fillVector(vectors.get(ci), df.column(ci), n);
            }

            writer.writeBatch();
            writer.end();
        }
    }
}
