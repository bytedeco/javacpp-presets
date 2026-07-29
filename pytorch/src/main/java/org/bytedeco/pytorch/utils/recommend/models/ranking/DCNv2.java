/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DCNv2.scala
 *
 * Deep & Cross Network V2. Reference: Stanford/Huawei.
 * Uses CrossNetMix (default) or CrossNetV2 for the cross path.
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Features;
import org.bytedeco.pytorch.utils.recommend.basic.layers.CrossNetMix;
import org.bytedeco.pytorch.utils.recommend.basic.layers.CrossNetV2;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DCNv2 extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long sparseDim;
    private final boolean useCrossNetMix;
    private final EmbeddingLayer embeddingLayer;
    private final CrossNetMix crossNetMix;   // one of these is non-null
    private final CrossNetV2 crossNetV2;
    private final MLP mlp;
    private final LinearImpl combo;

    public DCNv2(List<? extends Feature> features) {
        this(features, 8, 3, true, 4, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public DCNv2(List<? extends Feature> features, int embedDim, int numCrossLayers,
                 boolean useCrossNetMix, int lowRank, long[] mlpDims,
                 float dropout, String device) {
        super("DCNv2");
        this.useCrossNetMix = useCrossNetMix;
        List<Feature> featList = new ArrayList<>(features);

        this.embeddingLayer = new EmbeddingLayer(featList, embedDim, device);
        register_module("embedding", embeddingLayer);

        this.sparseDim = Features.calcSparseDim(featList);

        if (useCrossNetMix) {
            this.crossNetMix = new CrossNetMix(sparseDim, numCrossLayers, lowRank, 4, device);
            register_module("crossNet", crossNetMix);
            this.crossNetV2 = null;
        } else {
            this.crossNetV2 = new CrossNetV2(sparseDim, numCrossLayers, device);
            register_module("crossNet", crossNetV2);
            this.crossNetMix = null;
        }

        this.mlp = new MLP(sparseDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);

        this.combo = new LinearImpl(sparseDim + 1, 1);
        register_module("combo", combo);

        // Move entire model (embedding + cross + mlp + combo) so free params share device.
        if (device != null) {
            this.to(new Device(device), false);
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings = embeddingLayer.forward(sparseFeats);
        if (embeddings == null || embeddings.isNull() || !embeddings.defined()) {
            throw new IllegalStateException("DCNv2: embeddingLayer returned undefined tensor");
        }

        Tensor crossOut;
        if (useCrossNetMix) {
            crossOut = crossNetMix.forward(embeddings);
        } else {
            crossOut = crossNetV2.forward(embeddings);
        }
        if (crossOut == null || crossOut.isNull() || !crossOut.defined()) {
            throw new IllegalStateException("DCNv2: cross network returned undefined tensor");
        }

        Tensor deepOut = mlp.forward(embeddings);
        if (deepOut == null || deepOut.isNull() || !deepOut.defined()) {
            throw new IllegalStateException("DCNv2: mlp returned undefined tensor");
        }

        // new TensorVector(n) pre-fills n empty Tensors — must put(), not push_back().
        Tensor combined = TensorHelpers.cat(new Tensor[]{crossOut, deepOut}, 1);
        return combo.forward(combined);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
