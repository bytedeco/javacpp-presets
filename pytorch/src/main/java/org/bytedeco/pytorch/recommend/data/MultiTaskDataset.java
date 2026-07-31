/*
 * Ported from torch-rechub-scala: torchrec/data/Dataset.scala (MultiTaskDataset)
 *
 * Multi-task learning dataset: shared features + per-task labels.
 * Extends native Dataset via RecommendDataset.
 */
package org.bytedeco.pytorch.recommend.data;

import org.bytedeco.pytorch.SizeTOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.c10.SizeTArrayRef;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.data.ExampleVector;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.recommend.TensorHelpers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

//@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MultiTaskDataset extends RecommendDataset {

    private final Map<String, Tensor> features;
    private final Map<String, Tensor> taskLabels;
    private final long n;
    private final List<String> featureKeys;
    private final List<String> taskKeys;

    public MultiTaskDataset(Map<String, Tensor> features, Map<String, Tensor> taskLabels) {
        super();
        this.features = features != null ? new LinkedHashMap<>(features) : new LinkedHashMap<>();
        this.taskLabels = taskLabels != null ? new LinkedHashMap<>(taskLabels) : new LinkedHashMap<>();
        this.featureKeys = new ArrayList<>(this.features.keySet());
        Collections.sort(this.featureKeys);
        this.taskKeys = new ArrayList<>(this.taskLabels.keySet());
        Collections.sort(this.taskKeys);

        long size = 0L;
        if (!this.features.isEmpty()) {
            size = this.features.values().iterator().next().size(0);
        } else if (!this.taskLabels.isEmpty()) {
            size = this.taskLabels.values().iterator().next().size(0);
        }
        this.n = size;
    }

    @Override
    public long sizeLong() {
        return n;
    }

    @Override
    public Batch getBatch(long index) {
        Map<String, Tensor> tasks = rowMap(taskLabels, index);
        // Primary label = first task (stable sorted order) for native Example target
        Tensor primary = null;
        if (!taskKeys.isEmpty()) {
            primary = tasks.get(taskKeys.get(0));
        }
        return new Batch(
                rowMap(features, index),
                Collections.emptyMap(),
                Collections.emptyMap(),
                primary,
                null, null, null, null,
                Collections.emptyMap(),
                null,
                tasks);
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

    @Override
    public List<String> sparseOrder() {
        return featureKeys;
    }

    @Override
    public Example toExample(Batch batch) {
        Example base = super.toExample(batch, featureKeys, Collections.emptyList(), true);
        // If multiple tasks, pack all task labels into target as 1-D
        if (batch.taskLabels != null && batch.taskLabels.size() > 1) {
            List<Float> labs = new ArrayList<>();
            for (String k : taskKeys) {
                Tensor t = batch.taskLabels.get(k);
                if (t != null) {
                    float[] a = TensorHelpers.toFloatArray(t.toType(ScalarType.Float));
                    labs.add(a.length > 0 ? a[0] : 0f);
                } else {
                    labs.add(0f);
                }
            }
            float[] arr = new float[labs.size()];
            for (int i = 0; i < labs.size(); i++) arr[i] = labs.get(i);
            Tensor target = TensorHelpers.tensor(arr, arr.length);
            return new Example(base.data(), target);
        }
        return base;
    }

    public Map<String, Tensor> features() { return Collections.unmodifiableMap(features); }
    public Map<String, Tensor> taskLabels() { return Collections.unmodifiableMap(taskLabels); }
    public List<String> taskNames() { return Collections.unmodifiableList(taskKeys); }
}
