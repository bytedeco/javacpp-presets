/*
 * Ported from torch-rechub-scala: torchrec/data/Dataset.scala (Batch)
 *
 * A single training/inference sample (or mini-batch row).
 */
package org.bytedeco.pytorch.utils.recommend.data;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Batch {
    public final Map<String, Tensor> sparseFeatures;
    public final Map<String, Tensor> denseFeatures;
    public final Map<String, Tensor> sequenceFeatures;
    public final Tensor labels; // nullable
    public final Tensor tokens; // nullable
    public final Tensor positions; // nullable
    public final Tensor timeDiffs; // nullable
    public final Tensor targets; // nullable
    public final Map<String, Tensor> itemFeatures;
    public final Map<String, Tensor> negItemFeatures; // nullable map
    public final Map<String, Tensor> taskLabels; // nullable map

    public Batch(Map<String, Tensor> sparseFeatures) {
        this(sparseFeatures, Collections.emptyMap(), Collections.emptyMap(),
                null, null, null, null, null,
                Collections.emptyMap(), null, null);
    }

    public Batch(
            Map<String, Tensor> sparseFeatures,
            Map<String, Tensor> denseFeatures,
            Map<String, Tensor> sequenceFeatures,
            Tensor labels) {
        this(sparseFeatures, denseFeatures, sequenceFeatures, labels,
                null, null, null, null, Collections.emptyMap(), null, null);
    }

    public Batch(
            Map<String, Tensor> sparseFeatures,
            Map<String, Tensor> denseFeatures,
            Map<String, Tensor> sequenceFeatures,
            Tensor labels,
            Tensor tokens,
            Tensor positions,
            Tensor timeDiffs,
            Tensor targets,
            Map<String, Tensor> itemFeatures,
            Map<String, Tensor> negItemFeatures,
            Map<String, Tensor> taskLabels) {
        this.sparseFeatures = sparseFeatures != null ? sparseFeatures : Collections.emptyMap();
        this.denseFeatures = denseFeatures != null ? denseFeatures : Collections.emptyMap();
        this.sequenceFeatures = sequenceFeatures != null ? sequenceFeatures : Collections.emptyMap();
        this.labels = labels;
        this.tokens = tokens;
        this.positions = positions;
        this.timeDiffs = timeDiffs;
        this.targets = targets;
        this.itemFeatures = itemFeatures != null ? itemFeatures : Collections.emptyMap();
        this.negItemFeatures = negItemFeatures;
        this.taskLabels = taskLabels;
    }

    public Batch to(String device) {
        Device d = new Device(device);
        return new Batch(
                moveMap(sparseFeatures, d),
                moveMap(denseFeatures, d),
                moveMap(sequenceFeatures, d),
                labels != null ? labels.to(d, labels.dtype()) : null,
                tokens != null ? tokens.to(d, tokens.dtype()) : null,
                positions != null ? positions.to(d, positions.dtype()) : null,
                timeDiffs != null ? timeDiffs.to(d, timeDiffs.dtype()) : null,
                targets != null ? targets.to(d, targets.dtype()) : null,
                moveMap(itemFeatures, d),
                negItemFeatures != null ? moveMap(negItemFeatures, d) : null,
                taskLabels != null ? moveMap(taskLabels, d) : null);
    }

    public long numSamples() {
        if (!sparseFeatures.isEmpty()) {
            return sparseFeatures.values().iterator().next().size(0);
        }
        if (!denseFeatures.isEmpty()) {
            return denseFeatures.values().iterator().next().size(0);
        }
        if (!sequenceFeatures.isEmpty()) {
            return sequenceFeatures.values().iterator().next().size(0);
        }
        if (tokens != null) {
            return tokens.size(0);
        }
        return 0;
    }

    private static Map<String, Tensor> moveMap(Map<String, Tensor> map, Device d) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : map.entrySet()) {
            out.put(e.getKey(), e.getValue().to(d, e.getValue().dtype()));
        }
        return out;
    }

    /** Pack batch sparse/dense feature scalars into a Long index tensor. */
    public static Tensor packBatchToIndexTensor(
            Batch batch, List<String> sparseOrder, List<String> denseOrder, boolean includeLabel) {
        float[] sparseVals = new float[sparseOrder.size()];
        for (int i = 0; i < sparseOrder.size(); i++) {
            Tensor t = batch.sparseFeatures.get(sparseOrder.get(i));
            if (t != null) {
                float[] arr = TensorHelpers.toFloatArray(t);
                sparseVals[i] = arr.length > 0 ? arr[0] : 0.0f;
            }
        }
        float[] denseVals = new float[denseOrder != null ? denseOrder.size() : 0];
        if (denseOrder != null) {
            for (int i = 0; i < denseOrder.size(); i++) {
                Tensor t = batch.denseFeatures.get(denseOrder.get(i));
                if (t != null) {
                    float[] arr = TensorHelpers.toFloatArray(t);
                    denseVals[i] = arr.length > 0 ? arr[0] : 0.0f;
                }
            }
        }
        float[] labelVals = new float[0];
        if (includeLabel) {
            if (batch.labels != null) {
                float[] arr = TensorHelpers.toFloatArray(batch.labels);
                labelVals = new float[]{arr.length > 0 ? arr[0] : 0.0f};
            } else {
                labelVals = new float[]{0.0f};
            }
        }
        float[] vals = new float[sparseVals.length + denseVals.length + labelVals.length];
        System.arraycopy(sparseVals, 0, vals, 0, sparseVals.length);
        System.arraycopy(denseVals, 0, vals, sparseVals.length, denseVals.length);
        System.arraycopy(labelVals, 0, vals, sparseVals.length + denseVals.length, labelVals.length);
        if (vals.length == 0) {
            return org.bytedeco.pytorch.global.torch.zeros(new long[]{1L});
        }
        Tensor ft = TensorHelpers.tensor(vals, new long[]{vals.length});
        return ft.toType(ScalarType.Long);
    }
}
