package org.bytedeco.pytorch.dataframe.dataset;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.data.ExampleVector;
import org.bytedeco.pytorch.data.TensorExample;
import org.bytedeco.pytorch.data.TensorExampleVector;
import org.bytedeco.pytorch.global.torch;

/**
 * Helpers to bridge native DataLoader batches ({@link ExampleVector} of unstacked
 * examples) into stacked tensors matching {@link DataFrameDataLoader.Batch}.
 *
 * <p>Native {@code RandomDataLoader}/{@code SequentialDataLoader} over a
 * virtualized {@link org.bytedeco.pytorch.data.Dataset} yield
 * {@code ExampleVector} (no {@code map(ExampleStack)} binding for javacpp
 * Dataset). Stack here in Java.
 */
public final class NativeBatchSupport {

    private NativeBatchSupport() {}

    /** Stack data tensors of an {@link ExampleVector} on dim 0 → {@code [B, ...]}. */
    public static Tensor stackData(ExampleVector batch) {
        if (batch == null || batch.size() == 0) {
            return torch.empty(new long[]{0});
        }
        TensorVector tv = new TensorVector();
        long n = batch.size();
        for (long i = 0; i < n; i++) {
            tv.push_back(batch.get(i).data());
        }
        return torch.stack(tv, 0);
    }

    /** Stack target tensors of an {@link ExampleVector} on dim 0. */
    public static Tensor stackTarget(ExampleVector batch) {
        if (batch == null || batch.size() == 0) {
            return torch.empty(new long[]{0});
        }
        TensorVector tv = new TensorVector();
        long n = batch.size();
        for (long i = 0; i < n; i++) {
            tv.push_back(batch.get(i).target());
        }
        return torch.stack(tv, 0);
    }

    /** Stack both data and target into a single {@link Example}. */
    public static Example stack(ExampleVector batch) {
        return new Example(stackData(batch), stackTarget(batch));
    }

    /** Stack data tensors of a {@link TensorExampleVector}. */
    public static Tensor stackData(TensorExampleVector batch) {
        if (batch == null || batch.size() == 0) {
            return torch.empty(new long[]{0});
        }
        TensorVector tv = new TensorVector();
        long n = batch.size();
        for (long i = 0; i < n; i++) {
            tv.push_back(batch.get(i).data());
        }
        return torch.stack(tv, 0);
    }

    /** Number of examples in a native batch vector. */
    public static int batchSize(ExampleVector batch) {
        return batch == null ? 0 : (int) batch.size();
    }

    public static int batchSize(TensorExampleVector batch) {
        return batch == null ? 0 : (int) batch.size();
    }

    /**
     * Materialize one sample's primary feature tensor for a native view.
     * Package-visible helper used by adapters.
     */
    static Tensor primaryData(DataFrameDataset ds, NativeViewOptions opts, long index) {
        DataFrameDataset.Sample s = ds.get(index);
        return resolvePrimary(ds, opts, s);
    }

    static Tensor resolvePrimary(DataFrameDataset ds, NativeViewOptions opts, DataFrameDataset.Sample s) {
        NativeViewOptions.Mode mode = opts == null ? NativeViewOptions.Mode.AUTO : opts.mode();
        String primary = opts == null ? null : opts.primaryFeature();

        switch (mode) {
            case PRIMARY:
                if (primary == null || primary.isEmpty()) {
                    throw new IllegalArgumentException("PRIMARY mode requires primaryFeature name");
                }
                if ("__stacked__".equals(primary)) {
                    Tensor t = s.data();
                    return t != null ? t : torch.empty(new long[]{0});
                }
                return s.feature(primary);
            case STACKED_SCALARS:
                if (ds.scalarFeatureCount() > 0) {
                    Tensor t = s.data();
                    return t != null ? t : torch.empty(new long[]{0});
                }
                // fall through if no scalars
            case FIRST_SEQUENCE: {
                String[] seqs = ds.sequenceFeatureNames();
                if (seqs.length > 0) return s.feature(seqs[0]);
                Tensor t = s.data();
                return t != null ? t : torch.empty(new long[]{0});
            }
            case AUTO:
            default: {
                if (ds.scalarFeatureCount() > 0) {
                    Tensor t = s.data();
                    return t != null ? t : torch.empty(new long[]{0});
                }
                String[] seqs = ds.sequenceFeatureNames();
                if (seqs.length > 0) return s.feature(seqs[0]);
                Tensor t = s.data();
                return t != null ? t : torch.empty(new long[]{0});
            }
        }
    }

    static Tensor resolveTarget(DataFrameDataset ds, NativeViewOptions opts, DataFrameDataset.Sample s) {
        Tensor labels = s.labels();
        if (labels != null) return labels;
        boolean emptyOk = opts == null || opts.emptyTargetIfMissing();
        if (emptyOk) return torch.empty(new long[]{0});
        throw new IllegalStateException("Dataset has no labels and emptyTargetIfMissing=false");
    }

    /**
     * Build primary feature tensor for a batch of indices without going through Sample map.
     * Uses packed gather paths for efficiency.
     */
    static Tensor gatherPrimary(DataFrameDataset ds, NativeViewOptions opts, int[] idx) {
        NativeViewOptions.Mode mode = opts == null ? NativeViewOptions.Mode.AUTO : opts.mode();
        String primary = opts == null ? null : opts.primaryFeature();
        int B = idx.length;

        switch (mode) {
            case PRIMARY:
                if (primary == null || primary.isEmpty()) {
                    throw new IllegalArgumentException("PRIMARY mode requires primaryFeature name");
                }
                if ("__stacked__".equals(primary)) {
                    return gatherStackedScalars(ds, idx);
                }
                // named scalar or sequence
                for (String n : ds.scalarFeatureNames()) {
                    if (n.equals(primary)) {
                        // single scalar column as [B]
                        float[] all = ds.gatherScalars(idx);
                        int nFeat = ds.scalarFeatureCount();
                        int col = -1;
                        String[] names = ds.scalarFeatureNames();
                        for (int j = 0; j < names.length; j++) if (names[j].equals(primary)) { col = j; break; }
                        float[] colData = new float[B];
                        for (int b = 0; b < B; b++) colData[b] = all[b * nFeat + col];
                        return torch.tensor(colData);
                    }
                }
                // sequence
                return gatherSequenceTensor(ds, primary, idx);
            case STACKED_SCALARS:
                if (ds.scalarFeatureCount() > 0) return gatherStackedScalars(ds, idx);
                // fallthrough
            case FIRST_SEQUENCE: {
                String[] seqs = ds.sequenceFeatureNames();
                if (seqs.length > 0) return gatherSequenceTensor(ds, seqs[0], idx);
                if (ds.scalarFeatureCount() > 0) return gatherStackedScalars(ds, idx);
                return torch.empty(new long[]{B, 0});
            }
            case AUTO:
            default: {
                if (ds.scalarFeatureCount() > 0) return gatherStackedScalars(ds, idx);
                String[] seqs = ds.sequenceFeatureNames();
                if (seqs.length > 0) return gatherSequenceTensor(ds, seqs[0], idx);
                return torch.empty(new long[]{B, 0});
            }
        }
    }

    private static Tensor gatherStackedScalars(DataFrameDataset ds, int[] idx) {
        int B = idx.length;
        int nFeat = ds.scalarFeatureCount();
        if (nFeat == 0) return torch.empty(new long[]{B, 0});
        float[] batch = ds.gatherScalars(idx);
        return torch.tensor(batch).reshape(new long[]{B, nFeat});
    }

    private static Tensor gatherSequenceTensor(DataFrameDataset ds, String name, int[] idx) {
        int B = idx.length;
        int dim = ds.sequenceDim(name);
        Object packed = ds.gatherSequence(name, idx);
        if (packed instanceof long[]) {
            return torch.tensor((long[]) packed).reshape(new long[]{B, dim});
        }
        return torch.tensor((float[]) packed).reshape(new long[]{B, dim});
    }

    static Tensor gatherTarget(DataFrameDataset ds, NativeViewOptions opts, int[] idx) {
        int B = idx.length;
        int nLab = ds.labelCount();
        if (nLab == 0) {
            boolean emptyOk = opts == null || opts.emptyTargetIfMissing();
            if (emptyOk) return torch.empty(new long[]{B, 0});
            throw new IllegalStateException("Dataset has no labels and emptyTargetIfMissing=false");
        }
        Object packed = ds.gatherLabels(idx);
        if (ds.labelsAsLong()) {
            long[] data = (long[]) packed;
            return nLab == 1
                ? torch.tensor(data)
                : torch.tensor(data).reshape(new long[]{B, nLab});
        }
        float[] data = (float[]) packed;
        return nLab == 1
            ? torch.tensor(data)
            : torch.tensor(data).reshape(new long[]{B, nLab});
    }

    /** Split stacked batch tensor into per-row tensors for ExampleVector construction. */
    static ExampleVector toExampleVector(Tensor dataBatch, Tensor targetBatch) {
        long B = dataBatch.size(0);
        ExampleVector out = new ExampleVector(B);
        for (long i = 0; i < B; i++) {
            Tensor d = dataBatch.select(0, i).contiguous();
            Tensor t = targetBatch.size(0) == B
                ? targetBatch.select(0, i).contiguous()
                : torch.empty(new long[]{0});
            out.put(i, new Example(d, t));
        }
        return out;
    }

    static TensorExampleVector toTensorExampleVector(Tensor dataBatch) {
        long B = dataBatch.size(0);
        TensorExampleVector out = new TensorExampleVector(B);
        for (long i = 0; i < B; i++) {
            Tensor d = dataBatch.select(0, i).contiguous();
            out.put(i, new TensorExample(d));
        }
        return out;
    }
}
