/*
 * Ported from torch-rechub-scala: torchrec/models/matching/NCF.scala
 *
 * Neural Collaborative Filtering (NCF).
 * GMF + MLP fusion. Reference: He et al., WWW 2017
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NCF extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int userFieldIdx;
    private final int itemFieldIdx;
    private final int embedDim;
    private final int numFields;
    private final EmbeddingLayer embeddingLayer;
    private final MLP mlp;
    private final LinearImpl finalLinear;

    public NCF(List<? extends Feature> features) {
        this(features, 0, 1, 8, new long[]{64L, 32L}, 0.2f, DeviceSupport.backend());
    }

    public NCF(List<? extends Feature> features, int userFieldIdx, int itemFieldIdx,
               int embedDim, long[] mlpDims, float dropout, String device) {
        super("NCF");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (userFieldIdx == itemFieldIdx) {
            throw new IllegalArgumentException("userFieldIdx and itemFieldIdx must be different");
        }
        if (embedDim <= 0) {
            throw new IllegalArgumentException("embedDim must be positive");
        }
        this.userFieldIdx = userFieldIdx;
        this.itemFieldIdx = itemFieldIdx;
        this.embedDim = embedDim;

        List<Feature> featList = new ArrayList<>(features);
        this.embeddingLayer = new EmbeddingLayer(featList, embedDim, device);
        register_module("embedding", embeddingLayer);

        int nf = 0;
        for (Feature f : featList) {
            if (f instanceof SparseFeature) nf++;
        }
        this.numFields = nf;
        // GMF uses user/item elementwise product (embedDim).
        // MLP tower concatenates user+item embeddings → 2 * embedDim.
        // (Older ports incorrectly sized MLP with totalSparseDim = numFields*embedDim.)
        long mlpInDim = 2L * embedDim;
        long lastMlp = mlpDims[mlpDims.length - 1];
        this.mlp = new MLP(mlpInDim, mlpDims, lastMlp, "relu", dropout, false, device);
        register_module("mlp", mlp);

        long finalDim = embedDim + lastMlp;
        this.finalLinear = new LinearImpl(finalDim, 1);
        register_module("final_linear", finalLinear);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            mlp.to(dev, false);
            finalLinear.to(dev, false);
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings = embeddingLayer.forward(sparseFeats);
        int batchSize = (int) embeddings.size(0);
        Tensor emb3D = embeddings.view(batchSize, numFields, embedDim);

        Tensor userEmb = emb3D.select(1, userFieldIdx);
        Tensor itemEmb = emb3D.select(1, itemFieldIdx);

        // GMF: element-wise product
        Tensor gmfOut = userEmb.mul(itemEmb);

        // MLP: concatenate user and item embeddings
        // Note: Scala uses totalSparseDim for MLP input; concat is 2*embedDim.
        // Mirror Scala body literally (may require totalSparseDim == 2*embedDim at call sites).
        TensorVector cVec = new TensorVector();
        cVec.push_back(userEmb);
        cVec.push_back(itemEmb);
        Tensor concatEmb = torch.cat(cVec, 1);
        Tensor mlpOut = mlp.forward(concatEmb);

        TensorVector fVec = new TensorVector();
        fVec.push_back(gmfOut);
        fVec.push_back(mlpOut);
        Tensor combined = torch.cat(fVec, 1);

        return finalLinear.forward(combined).squeeze(1);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }

    public Tensor userForward(Map<String, Tensor> sparseFeats) {
        Tensor embeddings = embeddingLayer.forward(sparseFeats);
        int batchSize = (int) embeddings.size(0);
        Tensor emb3D = embeddings.view(batchSize, numFields, embedDim);
        return emb3D.select(1, userFieldIdx);
    }

    public Tensor itemForward(Map<String, Tensor> sparseFeats) {
        Tensor embeddings = embeddingLayer.forward(sparseFeats);
        int batchSize = (int) embeddings.size(0);
        Tensor emb3D = embeddings.view(batchSize, numFields, embedDim);
        return emb3D.select(1, itemFieldIdx);
    }
}
