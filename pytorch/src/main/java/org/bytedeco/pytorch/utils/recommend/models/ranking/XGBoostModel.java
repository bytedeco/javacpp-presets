/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/XGBoostModel.scala
 *
 * XGBoost-style ensemble of soft decision trees over embedded features.
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class XGBoostModel extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> features;
    private final long linkFeatDim;
    private final Device targetDevice;
    private final EmbeddingLayer embeddingLayer;
    private final List<SoftDecisionTree> trees = new ArrayList<>();

    public XGBoostModel(List<? extends Feature> features) {
        this(features, 64, 6, 8, 128L, DeviceSupport.backend());
    }

    public XGBoostModel(List<? extends Feature> features, int numTrees, int treeDepth,
                        int embedDim, long linkFeatDim, String device) {
        super("XGBoostModel");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (treeDepth < 1) {
            throw new IllegalArgumentException("treeDepth must be >= 1, got " + treeDepth);
        }
        if (linkFeatDim <= 0) {
            throw new IllegalArgumentException("linkFeatDim must be > 0, got " + linkFeatDim);
        }
        this.features = new ArrayList<>(features);
        this.linkFeatDim = linkFeatDim;
        this.targetDevice = new Device(device);

        int numLeaves = 1 << treeDepth;

        this.embeddingLayer = new EmbeddingLayer(this.features, embedDim, device);
        register_module("embedding", embeddingLayer);

        for (int i = 0; i < numTrees; i++) {
            SoftDecisionTree tree = new SoftDecisionTree(linkFeatDim, treeDepth, numLeaves, device);
            register_module("tree_" + i, tree);
            trees.add(tree);
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        int batchSize = (int) sparseFeats.values().iterator().next().size(0);

        // Filter to sparse feature names only
        Set<String> sparseFeatureNames = new HashSet<>();
        for (Feature f : features) {
            if (f instanceof SparseFeature) {
                sparseFeatureNames.add(f.name());
            }
        }
        java.util.Map<String, Tensor> validSparseFeats = new java.util.LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : sparseFeats.entrySet()) {
            if (sparseFeatureNames.contains(e.getKey())) {
                validSparseFeats.put(e.getKey(), e.getValue());
            }
        }

        Tensor embeddings = embeddingLayer.forward(validSparseFeats, Collections.emptyMap(), false);
        Tensor sparseFlat = embeddings.view(batchSize, -1);

        // Merge dense features
        Tensor input;
        if (denseFeats != null && !denseFeats.isEmpty()) {
            List<Tensor> denseSeq = new ArrayList<>(denseFeats.values());
            Tensor denseCat;
            if (denseSeq.size() == 1) {
                denseCat = denseSeq.get(0);
            } else {
                TensorVector dVec = new TensorVector();
                for (Tensor t : denseSeq) dVec.push_back(t);
                denseCat = torch.cat(dVec, 1L);
            }
            TensorVector cVec = new TensorVector();
            cVec.push_back(sparseFlat);
            cVec.push_back(denseCat);
            input = torch.cat(cVec, 1L);
        } else {
            input = sparseFlat;
        }

        // Dimension align to linkFeatDim
        int featDim = (int) input.size(1);
        Tensor alignedInput;
        if (featDim < linkFeatDim) {
            Tensor pad = torch.zeros(new long[]{batchSize, linkFeatDim - featDim},
                    new TensorOptions()
                            .device(new DeviceOptional(targetDevice))
                            .dtype(new ScalarTypeOptional(ScalarType.Float)));
            TensorVector pVec = new TensorVector();
            pVec.push_back(input);
            pVec.push_back(pad);
            alignedInput = torch.cat(pVec, 1L);
        } else if (featDim > linkFeatDim) {
            alignedInput = input.narrow(1, 0, linkFeatDim);
        } else {
            alignedInput = input;
        }

        // Average tree outputs
        TensorVector tVec = new TensorVector();
        for (SoftDecisionTree tree : trees) {
            tVec.push_back(tree.forward(alignedInput));
        }
        Tensor stacked = torch.stack(tVec, 0L);
        return stacked.mean(0L);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
