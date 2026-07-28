package org.bytedeco.pytorch.dataframe.dataset;

import java.util.Objects;

import org.bytedeco.pytorch.SizeTOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.c10.SizeTArrayRef;
import org.bytedeco.pytorch.data.TensorExample;
import org.bytedeco.pytorch.data.TensorExampleVector;
import org.bytedeco.pytorch.data.dataloader.RandomTensorDataLoader;
import org.bytedeco.pytorch.data.dataloader.SequentialTensorDataLoader;
import org.bytedeco.pytorch.data.datasets.JavaTensorDataset;

/**
 * Virtualized {@link JavaTensorDataset} adapter over a pure-Java
 * {@link DataFrameDataset} (features only — {@code TensorExample} / NoTarget).
 *
 * <p>Use with {@link RandomTensorDataLoader} / {@link SequentialTensorDataLoader}.
 * Labels are not exposed; use {@link DataFrameNativeDataset} for (data, target).
 *
 * <pre>
 *   DataFrameJavaTensorDataset tds = dfDs.asJavaTensorDataset();
 *   SequentialTensorDataLoader loader = dfDs.nativeTensorDataLoader()
 *       .batchSize(128).shuffle(false).buildSequential();
 * </pre>
 */
public class DataFrameJavaTensorDataset extends JavaTensorDataset {

    private final DataFrameDataset source;
    private final NativeViewOptions options;

    public DataFrameJavaTensorDataset(DataFrameDataset source) {
        this(source, NativeViewOptions.defaults());
    }

    public DataFrameJavaTensorDataset(DataFrameDataset source, NativeViewOptions options) {
        super();
        this.source = Objects.requireNonNull(source, "source");
        this.options = options == null ? NativeViewOptions.defaults() : options.copy();
    }

    public DataFrameDataset source() { return source; }

    public NativeViewOptions viewOptions() { return options; }

    @Override
    public TensorExample get(long index) {
        DataFrameDataset.Sample s = source.get(index);
        Tensor data = NativeBatchSupport.resolvePrimary(source, options, s);
        return new TensorExample(data);
    }

    @Override
    public SizeTOptional size() {
        return new SizeTOptional(source.sizeLong());
    }

    @Override
    public TensorExampleVector get_batch(SizeTArrayRef indices) {
        int n = (int) indices.size();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = (int) indices.get(i);
        }
        Tensor dataBatch = NativeBatchSupport.gatherPrimary(source, options, idx);
        return NativeBatchSupport.toTensorExampleVector(dataBatch);
    }

    public NativeTensorDataLoaderBuilder nativeTensorDataLoader() {
        return new NativeTensorDataLoaderBuilder(this);
    }

    public RandomTensorDataLoader randomTensorDataLoader(int batchSize) {
        return nativeTensorDataLoader().batchSize(batchSize).shuffle(true).buildRandom();
    }

    public SequentialTensorDataLoader sequentialTensorDataLoader(int batchSize) {
        return nativeTensorDataLoader().batchSize(batchSize).shuffle(false).buildSequential();
    }
}
