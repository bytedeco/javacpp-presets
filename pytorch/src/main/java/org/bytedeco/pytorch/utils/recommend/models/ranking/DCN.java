/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DCN.scala
 *
 * Deep & Cross Network. Reference: Stanford/Huawei.
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Features;
import org.bytedeco.pytorch.utils.recommend.basic.layers.CrossNetwork;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DCN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long sparseDim;
    private final EmbeddingLayer embeddingLayer;
    private final CrossNetwork crossNet;
    private final MLP mlp;
    private final LinearImpl combo;

    public DCN(List<? extends Feature> features) {
        this(features, 8, 3, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public DCN(List<? extends Feature> features, int embedDim, int numCrossLayers,
               long[] mlpDims, float dropout, String device) {
        super("DCN");
        List<Feature> featList = new ArrayList<>(features);
        this.embeddingLayer = new EmbeddingLayer(featList, embedDim, device);
        register_module("embedding", embeddingLayer);

        this.sparseDim = Features.calcSparseDim(featList);

        this.crossNet = new CrossNetwork(sparseDim, numCrossLayers, device);
        register_module("crossNet", crossNet);

        // Deep network: MLP outputs mlpDims.last dimension, not 1
        long lastDim = mlpDims[mlpDims.length - 1];
        this.mlp = new MLP(sparseDim, mlpDims, lastDim, "relu", dropout, false, device);
        register_module("mlp", mlp);

        this.combo = new LinearImpl(sparseDim + lastDim, 1);
        register_module("combo", combo);

        if (device != null && !"cpu".equals(device)) {
            combo.to(new Device(device), false);
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings3D = embeddingLayer.forward3D(sparseFeats, Collections.emptyMap());
        int batchSize = (int) embeddings3D.size(0);
        Tensor embeddings = embeddings3D.view(batchSize, (int) sparseDim);

        Tensor crossOut = crossNet.forward(embeddings);
        Tensor deepOut = mlp.forward(embeddings);

        TensorVector vec = new TensorVector();
        vec.push_back(crossOut);
        vec.push_back(deepOut);
        Tensor combined = torch.cat(vec, 1);
        return combo.forward(combined);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
