/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/AutoInt.scala
 *
 * AutoInt — Automatic Feature Interaction via Attentive Multi-Head Self-Attention.
 * Reference: https://arxiv.org/abs/1810.11921 (CIKM'2019).
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.InteractingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.LR;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AutoInt extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> sparseFeatures;
    private final List<Feature> denseFeatures;
    private final int numLayers;
    private final int embedDim;
    private final long dims;
    private final EmbeddingLayer sparseEmbedding;
    private final Map<String, LinearImpl> denseEmbeddings = new LinkedHashMap<>();
    private final List<InteractingLayer> interactingLayers = new ArrayList<>();
    private final LR linear;
    private final LinearImpl attnLinear;
    private final MLP mlp; // nullable

    public AutoInt(List<? extends Feature> sparseFeatures) {
        this(sparseFeatures, Collections.emptyList(), 2, 3, new long[]{128L, 64L},
                0.0f, true, DeviceSupport.backend());
    }

    public AutoInt(List<? extends Feature> sparseFeatures, List<? extends Feature> denseFeatures,
                   int numAttnHeads, int numLayers, long[] mlpDims, float dropout,
                   boolean useMlp, String device) {
        super("AutoInt");
        if (sparseFeatures == null || sparseFeatures.isEmpty()) {
            throw new IllegalArgumentException("AutoInt: sparseFeatures cannot be empty");
        }
        this.sparseFeatures = new ArrayList<>(sparseFeatures);
        this.denseFeatures = denseFeatures != null ? new ArrayList<>(denseFeatures) : new ArrayList<>();
        this.numLayers = numLayers;
        this.embedDim = this.sparseFeatures.get(0).embedDim();
        int numFields = this.sparseFeatures.size() + this.denseFeatures.size();
        this.dims = (long) numFields * embedDim;

        this.sparseEmbedding = new EmbeddingLayer(this.sparseFeatures, embedDim, device);
        register_module("sparse_embedding", sparseEmbedding);

        for (Feature fea : this.denseFeatures) {
            LinearImpl proj = new LinearImpl(new LinearOptions(1L, embedDim).bias(false));
            register_module("dense_" + fea.name(), proj);
            denseEmbeddings.put(fea.name(), proj);
        }

        for (int i = 0; i < numLayers; i++) {
            InteractingLayer layer = new InteractingLayer(embedDim, numAttnHeads, dropout, true, device);
            register_module("interacting_" + i, layer);
            interactingLayers.add(layer);
        }

        this.linear = new LR(dims, false, device);
        register_module("linear", linear);

        this.attnLinear = new LinearImpl(dims, 1L);
        register_module("attn_linear", attnLinear);

        if (useMlp) {
            this.mlp = new MLP(dims, mlpDims, 1L, "relu", dropout, false, device);
            register_module("mlp", mlp);
        } else {
            this.mlp = null;
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor sparseEmb = sparseEmbedding.forward3D(sparseFeats);

        List<Tensor> denseEmbList = new ArrayList<>();
        for (Feature fea : denseFeatures) {
            Tensor v = denseFeats.get(fea.name()).toType(ScalarType.Float).view(-1L, 1L, 1L);
            LinearImpl proj = denseEmbeddings.get(fea.name());
            denseEmbList.add(proj.forward(v));
        }

        Tensor embedX;
        if (!denseEmbList.isEmpty()) {
            TensorVector dVec = new TensorVector();
            for (Tensor t : denseEmbList) dVec.push_back(t);
            Tensor denseEmb = torch.cat(dVec, 1L);
            TensorVector cVec = new TensorVector();
            cVec.push_back(sparseEmb);
            cVec.push_back(denseEmb);
            embedX = torch.cat(cVec, 1L);
        } else {
            embedX = sparseEmb;
        }

        Tensor embedXFlat = embedX.view(embedX.size(0), -1L);

        Tensor attnOut = embedX;
        for (InteractingLayer layer : interactingLayers) {
            attnOut = layer.forward(attnOut);
        }

        Tensor attnOutFlat = attnOut.view(attnOut.size(0), -1L);
        Tensor yAttn = attnLinear.forward(attnOutFlat);
        Tensor yLinear = linear.forward(embedXFlat);

        Tensor y = yAttn.add(yLinear);
        if (mlp != null) {
            y = y.add(mlp.forward(embedXFlat));
        }
        return y;
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
