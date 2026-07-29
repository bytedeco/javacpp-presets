/*
 * Recommend-stack Dataset base.
 *
 * Extends the JavaCPP virtualized native Dataset (Example&lt;Tensor,Tensor&gt;) so
 * every concrete recommender dataset can be fed directly into
 * RandomDataLoader / SequentialDataLoader.
 *
 * Rich named features live in {@link Batch}; native {@link #get(long)} /
 * {@link #get_batch} pack them into a single data tensor + target for the
 * C++ DataLoader path.
 */
package org.bytedeco.pytorch.utils.recommend.data;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.SizeTOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.c10.SizeTArrayRef;
import org.bytedeco.pytorch.data.Dataset;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.data.ExampleVector;
import org.bytedeco.pytorch.data.dataloader.RandomDataLoader;
import org.bytedeco.pytorch.data.dataloader.SequentialDataLoader;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.data.sampler.RandomSampler;
import org.bytedeco.pytorch.data.sampler.SequentialSampler;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public abstract class RecommendDataset extends Dataset {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    protected RecommendDataset() {
        super(); // allocate native vtable for virtual get/size/get_batch
    }

    // ---- Scala-style API ----------------------------------------------------

    /** Number of samples (Java long, not SizeTOptional). */
    public abstract long sizeLong();

    /** Named-feature sample at {@code index}. */
    public abstract Batch getBatch(long index);

    // ---- Native Dataset contract --------------------------------------------

    @Override
    public Example get(long index) {
        Batch b = getBatch(index);
        return toExample(b);
    }

    @Override
    public SizeTOptional size() {
        return new SizeTOptional(sizeLong());
    }

    @Override
    public ExampleVector get_batch(SizeTArrayRef indices) {
        int n = (int) indices.size();
        ExampleVector vec = new ExampleVector(n);
        for (int i = 0; i < n; i++) {
            vec.put(i, get(indices.get(i)));
        }
        return vec;
    }

    // ---- Example packing ----------------------------------------------------

    /**
     * Pack a {@link Batch} into native {@link Example}:
     * <ul>
     *   <li>data   = concat(sparse[order], dense[order], optional seq flat) as Float 1-D</li>
     *   <li>target = label (or zeros) as Float 1-D / scalar</li>
     * </ul>
     * Subclasses may override for custom packing; default uses
     * {@link #sparseOrder()} / {@link #denseOrder()}.
     */
    public Example toExample(Batch batch) {
        return toExample(batch, sparseOrder(), denseOrder(), true);
    }

    public Example toExample(Batch batch, List<String> sparseOrder,
                             List<String> denseOrder, boolean includeLabel) {
        Tensor data = packFeatures(batch, sparseOrder, denseOrder);
        Tensor target;
        if (includeLabel && batch.labels != null) {
            target = batch.labels.toType(ScalarType.Float).reshape(-1L).contiguous().clone();
        } else {
            target = torch.zeros(new long[]{1L});
        }
        return new Example(data, target);
    }

    /**
     * Flatten ordered sparse (as float indices) + dense scalars into one 1-D float tensor.
     * Missing keys become 0.
     */
    public static Tensor packFeatures(Batch batch, List<String> sparseOrder, List<String> denseOrder) {
        List<Float> vals = new ArrayList<>();
        if (sparseOrder != null) {
            for (String name : sparseOrder) {
                Tensor t = batch.sparseFeatures.get(name);
                vals.add(scalarAsFloat(t));
            }
        }
        if (denseOrder != null) {
            for (String name : denseOrder) {
                Tensor t = batch.denseFeatures.get(name);
                vals.add(scalarAsFloat(t));
            }
        }
        // Also pack first sequence feature head if present and orders empty of dense/sparse
        if (vals.isEmpty() && batch.tokens != null) {
            float[] arr = TensorHelpers.toFloatArray(batch.tokens.toType(ScalarType.Float));
            for (float v : arr) vals.add(v);
        }
        if (vals.isEmpty()) {
            return torch.zeros(new long[]{1L});
        }
        float[] data = new float[vals.size()];
        for (int i = 0; i < vals.size(); i++) data[i] = vals.get(i);
        return TensorHelpers.tensor(data, data.length);
    }

    private static float scalarAsFloat(Tensor t) {
        if (t == null) return 0.0f;
        try {
            float[] arr = TensorHelpers.toFloatArray(t.toType(ScalarType.Float));
            return arr.length > 0 ? arr[0] : 0.0f;
        } catch (Throwable e) {
            return 0.0f;
        }
    }

    /** Stable sparse feature key order for packing (override in subclasses). */
    public List<String> sparseOrder() {
        return Collections.emptyList();
    }

    /** Stable dense feature key order for packing (override in subclasses). */
    public List<String> denseOrder() {
        return Collections.emptyList();
    }

    // ---- Native DataLoader factories ----------------------------------------

    public RandomDataLoader randomDataLoader(long batchSize) {
        return randomDataLoader(batchSize, 0L);
    }

    public RandomDataLoader randomDataLoader(long batchSize, long workers) {
        long n = sizeLong();
        DataLoaderOptions opts = new DataLoaderOptions(batchSize);
        if (workers > 0) opts.workers(workers);
        return new RandomDataLoader(this, new RandomSampler(n), opts);
    }

    public SequentialDataLoader sequentialDataLoader(long batchSize) {
        return sequentialDataLoader(batchSize, 0L);
    }

    public SequentialDataLoader sequentialDataLoader(long batchSize, long workers) {
        long n = sizeLong();
        DataLoaderOptions opts = new DataLoaderOptions(batchSize);
        if (workers > 0) opts.workers(workers);
        return new SequentialDataLoader(this, new SequentialSampler(n), opts);
    }

    // ---- Tensor slice helpers for subclasses --------------------------------

    /** Safe narrow along dim 0 with clone (owned storage). */
    protected static Tensor row(Tensor v, long index) {
        long n = v.size(0);
        long safe = Math.min(Math.max(index, 0L), Math.max(n - 1, 0L));
        Tensor sliced = v.narrow(0, safe, 1);
        if (sliced.dim() == 0) sliced = sliced.unsqueeze(0);
        return sliced.contiguous().clone();
    }

    protected static Map<String, Tensor> rowMap(Map<String, Tensor> map, long index) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : map.entrySet()) {
            out.put(e.getKey(), row(e.getValue(), index));
        }
        return out;
    }

    protected static Map<String, Tensor> narrowMap(Map<String, Tensor> map, long start, long length) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : map.entrySet()) {
            out.put(e.getKey(), e.getValue().narrow(0, start, length).contiguous().clone());
        }
        return out;
    }

    /** Build Long 1-D feature tensor from float indices. */
    public static Tensor longFeature(float[] data) {
        return TensorHelpers.tensor(data, data.length).toType(ScalarType.Long);
    }

    public static Tensor longFeature(long[] data) {
        return TensorHelpers.longTensorDirect(data);
    }

    public static Tensor floatFeature(float[] data) {
        return TensorHelpers.tensor(data, data.length);
    }
}
