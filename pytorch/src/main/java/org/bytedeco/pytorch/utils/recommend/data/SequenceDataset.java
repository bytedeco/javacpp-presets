/*
 * Ported from torch-rechub-scala: torchrec/data/Dataset.scala (SequenceDataset)
 *
 * Sequence / session dataset for sequential recommenders (DIN, BST, SASRec, ...).
 * Extends native Dataset via RecommendDataset.
 */
package org.bytedeco.pytorch.utils.recommend.data;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SequenceDataset extends RecommendDataset {

    private final Map<String, Tensor> features;
    private final Map<String, Tensor> sequenceFeatures;
    private final Tensor labels;
    private final Tensor positions;
    private final Tensor timeDiffs;
    private final Tensor tokens;
    private final Tensor targets;
    private final Map<String, Tensor> itemFeatures;
    private final long n;
    private final List<String> featureKeys;
    private final List<String> seqKeys;

    public SequenceDataset(
            Map<String, Tensor> features,
            Map<String, Tensor> sequenceFeatures,
            Tensor labels) {
        this(features, sequenceFeatures, labels, null, null, null, null, null);
    }

    public SequenceDataset(
            Map<String, Tensor> features,
            Map<String, Tensor> sequenceFeatures,
            Tensor labels,
            Tensor positions,
            Tensor timeDiffs,
            Tensor tokens,
            Tensor targets,
            Map<String, Tensor> itemFeatures) {
        super();
        this.features = features != null ? new LinkedHashMap<>(features) : new LinkedHashMap<>();
        this.sequenceFeatures = sequenceFeatures != null
                ? new LinkedHashMap<>(sequenceFeatures) : new LinkedHashMap<>();
        this.labels = labels;
        this.positions = positions;
        this.timeDiffs = timeDiffs;
        this.tokens = tokens;
        this.targets = targets;
        this.itemFeatures = itemFeatures;
        this.featureKeys = new ArrayList<>(this.features.keySet());
        Collections.sort(this.featureKeys);
        this.seqKeys = new ArrayList<>(this.sequenceFeatures.keySet());
        Collections.sort(this.seqKeys);

        long size = 0L;
        if (tokens != null) size = tokens.size(0);
        else if (!this.features.isEmpty())
            size = this.features.values().iterator().next().size(0);
        else if (!this.sequenceFeatures.isEmpty())
            size = this.sequenceFeatures.values().iterator().next().size(0);
        this.n = size;
    }

    @Override
    public long sizeLong() {
        return n;
    }

    @Override
    public Batch getBatch(long index) {
        return new Batch(
                rowMap(features, index),
                Collections.emptyMap(),
                rowMap(sequenceFeatures, index),
                labels != null ? row(labels, index) : null,
                tokens != null ? row(tokens, index) : null,
                positions != null ? row(positions, index) : null,
                timeDiffs != null ? row(timeDiffs, index) : null,
                targets != null ? row(targets, index) : null,
                itemFeatures != null ? rowMap(itemFeatures, index) : Collections.emptyMap(),
                null,
                null);
    }

    @Override
    public List<String> sparseOrder() {
        return featureKeys;
    }

    @Override
    public Example toExample(Batch batch) {
        // Pack: context features + flattened first sequence row + optional target
        List<Float> vals = new ArrayList<>();
        for (String k : featureKeys) {
            Tensor t = batch.sparseFeatures.get(k);
            if (t != null) {
                float[] a = TensorHelpers.toFloatArray(t.toType(ScalarType.Float));
                if (a.length > 0) vals.add(a[0]);
                else vals.add(0f);
            } else {
                vals.add(0f);
            }
        }
        for (String k : seqKeys) {
            Tensor t = batch.sequenceFeatures.get(k);
            if (t != null) {
                float[] a = TensorHelpers.toFloatArray(t.toType(ScalarType.Float));
                for (float v : a) vals.add(v);
            }
        }
        if (batch.tokens != null) {
            float[] a = TensorHelpers.toFloatArray(batch.tokens.toType(ScalarType.Float));
            for (float v : a) vals.add(v);
        }
        float[] dataArr = new float[Math.max(vals.size(), 1)];
        for (int i = 0; i < vals.size(); i++) dataArr[i] = vals.get(i);
        Tensor data = TensorHelpers.tensor(dataArr, dataArr.length);
        Tensor target = batch.labels != null
                ? batch.labels.toType(ScalarType.Float).reshape(-1L).contiguous().clone()
                : (batch.targets != null
                    ? batch.targets.toType(ScalarType.Float).reshape(-1L).contiguous().clone()
                    : org.bytedeco.pytorch.global.torch.zeros(new long[]{1L}));
        return new Example(data, target);
    }

    public Map<String, Tensor> features() { return Collections.unmodifiableMap(features); }
    public Map<String, Tensor> sequenceFeatures() { return Collections.unmodifiableMap(sequenceFeatures); }
    public Tensor labels() { return labels; }
    public Tensor tokens() { return tokens; }
    public Tensor targets() { return targets; }
}
