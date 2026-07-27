package org.bytedeco.pytorch.data.arrow;

import java.io.FileInputStream;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowFileReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.util.TransferPair;
import org.bytedeco.pytorch.JvmModuleSupport;
import org.bytedeco.pytorch.data.dataframe.ArrowStorage;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

/**
 * Local-only Arrow IPC / Feather v2 reader.
 * Prefers zero-copy {@link ArrowStorage} columns when a single record batch is present.
 */
public final class LocalArrowIpcReader {

    private LocalArrowIpcReader() {}

    public static DataFrame read(String path) throws Exception {
        JvmModuleSupport.ensureNioBufferAccess();
        try {
            return readFile(path);
        } catch (Exception fileEx) {
            try {
                return readStream(path);
            } catch (Exception streamEx) {
                fileEx.addSuppressed(streamEx);
                throw new Exception("Failed to read Arrow IPC from " + path, fileEx);
            }
        }
    }

    private static DataFrame readFile(String path) throws Exception {
        BufferAllocator alloc = new RootAllocator();
        try {
            try (FileChannel channel = FileChannel.open(Path.of(path), StandardOpenOption.READ);
                 ArrowFileReader reader = new ArrowFileReader(channel, alloc)) {
                List<VectorSchemaRoot> batches = new ArrayList<>();
                for (var block : reader.getRecordBlocks()) {
                    reader.loadRecordBatch(block);
                    // transfer ownership out of reader root
                    VectorSchemaRoot src = reader.getVectorSchemaRoot();
                    batches.add(transferRoot(src, alloc));
                }
                return buildFromBatches(batches, alloc);
            }
        } catch (Exception e) {
            alloc.close();
            throw e;
        }
    }

    private static DataFrame readStream(String path) throws Exception {
        BufferAllocator alloc = new RootAllocator();
        try {
            try (FileInputStream fis = new FileInputStream(path);
                 ArrowStreamReader reader = new ArrowStreamReader(fis, alloc)) {
                List<VectorSchemaRoot> batches = new ArrayList<>();
                while (reader.loadNextBatch()) {
                    batches.add(transferRoot(reader.getVectorSchemaRoot(), alloc));
                }
                return buildFromBatches(batches, alloc);
            }
        } catch (Exception e) {
            alloc.close();
            throw e;
        }
    }

    /** Deep-transfer vectors so they outlive the reader root. */
    private static VectorSchemaRoot transferRoot(VectorSchemaRoot src, BufferAllocator alloc) {
        List<FieldVector> transferred = new ArrayList<>();
        for (FieldVector sv : src.getFieldVectors()) {
            FieldVector dv = sv.getField().createVector(alloc);
            TransferPair tp = sv.makeTransferPair(dv);
            // copy data (transfer would clear source; splitAndTransfer / copy safer across batches)
            tp.splitAndTransfer(0, sv.getValueCount());
            dv.setValueCount(sv.getValueCount());
            transferred.add(dv);
        }
        VectorSchemaRoot root = new VectorSchemaRoot(transferred);
        root.setRowCount(src.getRowCount());
        return root;
    }

    private static DataFrame buildFromBatches(List<VectorSchemaRoot> batches, BufferAllocator alloc) {
        DataFrame df = DataFrame.create();
        df.addResource(alloc);
        if (batches.isEmpty()) {
            return df;
        }
        if (batches.size() == 1) {
            VectorSchemaRoot root = batches.get(0);
            df.addResource(root);
            for (FieldVector vec : root.getFieldVectors()) {
                Column.DType dtype = ArrowSchemaMapper.fromField(vec.getField());
                // Do not close vector with column — root owns it; ArrowStorage must not close vector
                Column col = new Column(vec.getName(), new ArrowStorage(dtype, vec, alloc, /*ownVector*/ false));
                df.addColumn(col);
            }
            // row count from vectors
            df.syncRowCountPublic();
            return df;
        }
        // multi-batch: materialize to list storage
        VectorSchemaRoot first = batches.get(0);
        for (FieldVector vec : first.getFieldVectors()) {
            Column.DType dtype = ArrowSchemaMapper.fromField(vec.getField());
            df.addColumn(vec.getName(), dtype);
        }
        for (VectorSchemaRoot root : batches) {
            int rows = root.getRowCount();
            List<FieldVector> vectors = root.getFieldVectors();
            for (int r = 0; r < rows; r++) {
                Object[] vals = new Object[vectors.size()];
                for (int c = 0; c < vectors.size(); c++) {
                    FieldVector v = vectors.get(c);
                    vals[c] = v.isNull(r) ? null : ArrowStorage.readValue(v, r, ArrowSchemaMapper.fromField(v.getField()));
                }
                df.addRow(vals);
            }
            try { root.close(); } catch (Exception ignored) {}
        }
        return df;
    }
}
