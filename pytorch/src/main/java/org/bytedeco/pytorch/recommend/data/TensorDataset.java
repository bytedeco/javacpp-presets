/*
 * Ported from torch-rechub-scala: torchrec/data/Dataset.scala (TensorDataset)
 *
 * In-memory ranking dataset: sparse + dense features + optional label.
 * Extends native {@link org.bytedeco.pytorch.data.Dataset} via RecommendDataset.
 */
package org.bytedeco.pytorch.recommend.data;

import org.bytedeco.pytorch.SizeTOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.c10.SizeTArrayRef;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.data.ExampleVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

//@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TensorDataset extends RecommendDataset {

    private final Map<String, Tensor> sparseFeatures;
    private final Map<String, Tensor> denseFeatures;
    private final Tensor labels; // nullable
    private final long n;
    private final List<String> sparseKeys;
    private final List<String> denseKeys;

    public TensorDataset(Map<String, Tensor> sparseFeatures) {
        this(sparseFeatures, Collections.emptyMap(), null);
    }

    public TensorDataset(
            Map<String, Tensor> sparseFeatures,
            Map<String, Tensor> denseFeatures,
            Tensor labels) {
        super();
        this.sparseFeatures = sparseFeatures != null
                ? new LinkedHashMap<>(sparseFeatures) : new LinkedHashMap<>();
        this.denseFeatures = denseFeatures != null
                ? new LinkedHashMap<>(denseFeatures) : new LinkedHashMap<>();
        this.labels = labels;
        this.sparseKeys = new ArrayList<>(this.sparseFeatures.keySet());
        Collections.sort(this.sparseKeys);
        this.denseKeys = new ArrayList<>(this.denseFeatures.keySet());
        Collections.sort(this.denseKeys);

        long size = 0L;
        if (!this.sparseFeatures.isEmpty()) {
            size = this.sparseFeatures.values().iterator().next().size(0);
        } else if (!this.denseFeatures.isEmpty()) {
            size = this.denseFeatures.values().iterator().next().size(0);
        } else if (labels != null) {
            size = labels.size(0);
        }
        this.n = size;
    }

    @Override
    public long sizeLong() {
        return n;
    }

    @Override
    public Batch getBatch(long index) {
        return new Batch(
                rowMap(sparseFeatures, index),
                rowMap(denseFeatures, index),
                Collections.emptyMap(),
                labels != null ? row(labels, index) : null);
    }

    @Override
    public List<String> sparseOrder() {
        return sparseKeys;
    }

    @Override
    public List<String> denseOrder() {
        return denseKeys;
    }

    public Map<String, Tensor> sparseFeatures() {
        return Collections.unmodifiableMap(sparseFeatures);
    }

    public Map<String, Tensor> denseFeatures() {
        return Collections.unmodifiableMap(denseFeatures);
    }

    public Tensor labels() {
        return labels;
    }

    /** Slice [start, start+length) into a new TensorDataset (owned clones). */
    public TensorDataset slice(long start, long length) {
        long s = Math.max(0L, start);
        long len = Math.min(length, Math.max(0L, n - s));
        Map<String, Tensor> sp = narrowMap(sparseFeatures, s, len);
        Map<String, Tensor> de = narrowMap(denseFeatures, s, len);
        Tensor lab = labels != null ? labels.narrow(0, s, len).contiguous().clone() : null;
        return new TensorDataset(sp, de, lab);
    }

    @Override
    public Example get(long index) {
        return super.get(index);
    }

    @Override
    public SizeTOptional size() {
        return super.size();
    }

    @Override
    public ExampleVector get_batch(SizeTArrayRef indices) {
        return super.get_batch(indices);
    }
}
