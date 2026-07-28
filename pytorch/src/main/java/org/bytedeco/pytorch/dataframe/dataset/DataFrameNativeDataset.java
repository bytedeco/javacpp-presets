package org.bytedeco.pytorch.dataframe.dataset;

import java.util.Objects;

import org.bytedeco.pytorch.SizeTOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.c10.SizeTArrayRef;
import org.bytedeco.pytorch.data.Dataset;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.data.ExampleVector;
import org.bytedeco.pytorch.data.dataloader.RandomDataLoader;
import org.bytedeco.pytorch.data.dataloader.SequentialDataLoader;
import org.bytedeco.pytorch.global.torch;

/**
 * Virtualized native {@link Dataset} adapter over a pure-Java {@link DataFrameDataset}.
 *
 * <p>Enables {@link RandomDataLoader} / {@link SequentialDataLoader} and any API that
 * requires {@code javacpp::Dataset&lt;Tensor,Tensor&gt;}. Named multi-feature access
 * remains on the source {@link DataFrameDataset} / {@link DataFrameDataLoader}.
 *
 * <pre>
 *   DataFrameNativeDataset nativeDs = dfDs.asDataset();
 *   SequentialDataLoader loader = dfDs.nativeDataLoader()
 *       .batchSize(256).shuffle(false).buildSequential();
 *   for (var it = loader.begin(); !it.equals(loader.end()); it = it.increment()) {
 *       Example batch = NativeBatchSupport.stack(it.access());
 *       // batch.data() [B, F], batch.target() [B] or [B, L]
 *   }
 * </pre>
 */
public class DataFrameNativeDataset extends Dataset {

    private final DataFrameDataset source;
    private final NativeViewOptions options;

    public DataFrameNativeDataset(DataFrameDataset source) {
        this(source, NativeViewOptions.defaults());
    }

    public DataFrameNativeDataset(DataFrameDataset source, NativeViewOptions options) {
        super(); // allocate native vtable
        this.source = Objects.requireNonNull(source, "source");
        this.options = options == null ? NativeViewOptions.defaults() : options.copy();
    }

    /** Underlying pure-Java dataset (packed features, named sequences). */
    public DataFrameDataset source() { return source; }

    public NativeViewOptions viewOptions() { return options; }

    @Override
    public Example get(long index) {
        DataFrameDataset.Sample s = source.get(index);
        Tensor data = NativeBatchSupport.resolvePrimary(source, options, s);
        Tensor target = NativeBatchSupport.resolveTarget(source, options, s);
        return new Example(data, target);
    }

    @Override
    public SizeTOptional size() {
        return new SizeTOptional(source.sizeLong());
    }

    @Override
    public ExampleVector get_batch(SizeTArrayRef indices) {
        int n = (int) indices.size();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = (int) indices.get(i);
        }
        // Efficient path: gather packed rows then split into ExampleVector
        Tensor dataBatch = NativeBatchSupport.gatherPrimary(source, options, idx);
        Tensor targetBatch = NativeBatchSupport.gatherTarget(source, options, idx);
        // If target is [B] and data is [B, F], select works; if target empty [B,0] still ok
        return NativeBatchSupport.toExampleVector(dataBatch, normalizeTargetBatch(targetBatch, n));
    }

    private static Tensor normalizeTargetBatch(Tensor targetBatch, int B) {
        if (targetBatch == null) return torch.empty(new long[]{B, 0});
        if (targetBatch.dim() == 0) {
            // scalar? unexpected — wrap
            return targetBatch.unsqueeze(0);
        }
        if (targetBatch.size(0) != B && targetBatch.numel() == 0) {
            return torch.empty(new long[]{B, 0});
        }
        return targetBatch;
    }

    /** Fluent native loader builder bound to this adapter. */
    public NativeDataLoaderBuilder nativeDataLoader() {
        return new NativeDataLoaderBuilder(this);
    }

    public RandomDataLoader randomDataLoader(int batchSize) {
        return nativeDataLoader().batchSize(batchSize).shuffle(true).buildRandom();
    }

    public SequentialDataLoader sequentialDataLoader(int batchSize) {
        return nativeDataLoader().batchSize(batchSize).shuffle(false).buildSequential();
    }
}
