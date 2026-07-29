/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/FiBiNet.scala
 *
 * FiBiNet: Feature Importance and Bilinear feature Interaction.
 * Reference: RecSys 2019
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.utils.recommend.basic.layers.SENETLayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FiBiNet extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer embeddingLayer;
    private final SENETLayer senet;
    private final BilinearInteraction bilinear;
    private final MLP mlp;
    private final int numFields;
    private final int embedDim;
    private final String bilinearType;

    public FiBiNet(List<? extends Feature> features) {
        this(features, 8, new long[]{256L, 128L}, 3, "field_interaction", 0.2f, DeviceSupport.backend());
    }

    public FiBiNet(List<? extends Feature> features, int embedDim, long[] mlpDims,
                   int reduction, String bilinearType, float dropout, String device) {
        super("FiBiNet");
        this.embedDim = embedDim;
        this.bilinearType = bilinearType != null ? bilinearType : "field_interaction";

        List<Feature> featList = new ArrayList<>(features);
        this.embeddingLayer = new EmbeddingLayer(featList, embedDim, device);
        register_module("embedding", embeddingLayer);

        int nf = 0;
        for (Feature f : featList) {
            if (f instanceof SparseFeature) nf++;
        }
        this.numFields = nf;

        this.senet = new SENETLayer(numFields, reduction, device);
        register_module("senet", senet);

        this.bilinear = new BilinearInteraction(embedDim, numFields, this.bilinearType, device);
        register_module("bilinear", bilinear);

        // MLP input dim depends on bilinearType:
        // - field_all / field_each: numFields^2 * embedDim (all pairs)
        // - field_interaction: numFields * (numFields-1) / 2 * embedDim (only i<j pairs)
        int numFieldPairs;
        if ("field_all".equals(this.bilinearType) || "field_each".equals(this.bilinearType)) {
            numFieldPairs = numFields * numFields;
        } else {
            numFieldPairs = numFields * (numFields - 1) / 2;
        }
        long mlpInputDim = (long) numFieldPairs * embedDim * 2; // both bi1 and bi2
        this.mlp = new MLP(mlpInputDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        // Use forward3D to get (batch, numFields, embedDim)
        Tensor embeddings = embeddingLayer.forward3D(sparseFeats);

        // SENET-enhanced features
        Tensor senetFeatures = senet.forward(embeddings);

        // Bilinear interactions on original embeddings
        List<Tensor> biOut1 = bilinear.forwardPair(embeddings, embeddings);

        // Bilinear interactions on SENET-enhanced embeddings
        List<Tensor> biOut2 = bilinear.forwardPair(senetFeatures, embeddings);

        // Stack then concatenate along dim=1
        TensorVector v1 = new TensorVector();
        for (Tensor t : biOut1) v1.push_back(t);
        Tensor stacked1 = torch.stack(v1, 1L);

        TensorVector v2 = new TensorVector();
        for (Tensor t : biOut2) v2.push_back(t);
        Tensor stacked2 = torch.stack(v2, 1L);

        TensorVector cVec = new TensorVector();
        cVec.push_back(stacked1);
        cVec.push_back(stacked2);
        Tensor combined = torch.cat(cVec, 1L);

        // Flatten and pass through MLP
        Tensor flattened = combined.view(combined.size(0), -1);
        return mlp.forward(flattened);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
