/*
 * Ported from torch-rechub-scala: torchrec/data/Dataset.scala (MatchingDataset)
 *
 * Two-tower / retrieval dataset: user features + item features + optional label
 * and negative item features. Extends native Dataset via RecommendDataset.
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
public class MatchingDataset extends RecommendDataset {

    private final Map<String, Tensor> userFeatures;
    private final Map<String, Tensor> itemFeatures;
    private final Tensor labels; // nullable
    private final Map<String, Tensor> negItemFeatures; // nullable
    private final Tensor tokens; // nullable
    private final Tensor positions; // nullable
    private final long n;
    private final List<String> userKeys;
    private final List<String> itemKeys;

    public MatchingDataset(
            Map<String, Tensor> userFeatures,
            Map<String, Tensor> itemFeatures) {
        this(userFeatures, itemFeatures, null, null, null, null);
    }

    public MatchingDataset(
            Map<String, Tensor> userFeatures,
            Map<String, Tensor> itemFeatures,
            Tensor labels) {
        this(userFeatures, itemFeatures, labels, null, null, null);
    }

    public MatchingDataset(
            Map<String, Tensor> userFeatures,
            Map<String, Tensor> itemFeatures,
            Tensor labels,
            Map<String, Tensor> negItemFeatures,
            Tensor tokens,
            Tensor positions) {
        super();
        this.userFeatures = userFeatures != null
                ? new LinkedHashMap<>(userFeatures) : new LinkedHashMap<>();
        this.itemFeatures = itemFeatures != null
                ? new LinkedHashMap<>(itemFeatures) : new LinkedHashMap<>();
        this.labels = labels;
        this.negItemFeatures = negItemFeatures;
        this.tokens = tokens;
        this.positions = positions;
        this.userKeys = new ArrayList<>(this.userFeatures.keySet());
        Collections.sort(this.userKeys);
        this.itemKeys = new ArrayList<>(this.itemFeatures.keySet());
        Collections.sort(this.itemKeys);

        long u = this.userFeatures.isEmpty() ? 0L
                : this.userFeatures.values().iterator().next().size(0);
        long it = this.itemFeatures.isEmpty() ? 0L
                : this.itemFeatures.values().iterator().next().size(0);
        this.n = Math.max(u, it);
    }

    @Override
    public long sizeLong() {
        return n;
    }

    @Override
    public Batch getBatch(long index) {
        return new Batch(
                rowMap(userFeatures, index),
                Collections.emptyMap(),
                Collections.emptyMap(),
                labels != null ? row(labels, index) : null,
                tokens != null ? row(tokens, index) : null,
                positions != null ? row(positions, index) : null,
                null,
                null,
                rowMap(itemFeatures, index),
                negItemFeatures != null ? rowMap(negItemFeatures, index) : null,
                null);
    }

    @Override
    public List<String> sparseOrder() {
        // Pack user then item features for native Example path
        List<String> order = new ArrayList<>(userKeys.size() + itemKeys.size());
        order.addAll(userKeys);
        // item keys are under itemFeatures in Batch; packFeatures only sees sparseFeatures
        // so we override toExample for matching.
        return order;
    }

    @Override
    public Example toExample(Batch batch) {
        // Concat user sparse + item features into data; label as target
        List<String> order = new ArrayList<>();
        order.addAll(userKeys);
        // Build a synthetic sparse map: user + item under flat keys
        Map<String, Tensor> flat = new LinkedHashMap<>();
        flat.putAll(batch.sparseFeatures);
        for (Map.Entry<String, Tensor> e : batch.itemFeatures.entrySet()) {
            flat.put("item_" + e.getKey(), e.getValue());
        }
        List<String> flatOrder = new ArrayList<>(flat.keySet());
        Collections.sort(flatOrder);
        Batch packed = new Batch(flat, batch.denseFeatures, batch.sequenceFeatures, batch.labels);
        return super.toExample(packed, flatOrder, denseOrder(), true);
    }

    public Map<String, Tensor> userFeatures() {
        return Collections.unmodifiableMap(userFeatures);
    }

    public Map<String, Tensor> itemFeatures() {
        return Collections.unmodifiableMap(itemFeatures);
    }

    public Tensor labels() {
        return labels;
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
