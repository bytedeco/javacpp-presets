/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/xDeepFM.scala
 *
 * xDeepFM: eXtreme Deep Factorization Machine.
 * Combines CIN + FM + DNN + Linear.
 * Reference: KDD 2018
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.CINFixed;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.FM;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class XDeepFM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final int embedDim;
    private final long sparseDim;
    private final EmbeddingLayer embeddingLayer;
    private final CINFixed cin;
    private final FM fmLayer;
    private final MLP mlp;
    private final LinearImpl linearWeight;

    public XDeepFM(List<? extends Feature> features) {
        this(features, 8, new int[]{128, 64}, new long[]{256L, 128L, 64L},
                true, 0.2f, DeviceSupport.backend());
    }

    public XDeepFM(List<? extends Feature> features, int embedDim, int[] crossLayerSizes,
                   long[] mlpDims, boolean splitHalf, float dropout, String device) {
        super("xDeepFM");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        this.embedDim = embedDim;

        int nf = 0;
        for (Feature f : features) {
            if (f instanceof SparseFeature) nf++;
        }
        if (nf < 2) {
            throw new IllegalArgumentException("xDeepFM requires at least 2 sparse features");
        }
        this.numFields = nf;
        this.sparseDim = (long) numFields * embedDim;

        this.embeddingLayer = new EmbeddingLayer(new ArrayList<>(features), embedDim, device);
        register_module("embedding", embeddingLayer);

        this.cin = new CINFixed(numFields, embedDim, crossLayerSizes, splitHalf, device);
        register_module("cin", cin);

        this.fmLayer = new FM(embedDim, device);
        register_module("fm", fmLayer);

        this.mlp = new MLP(sparseDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);

        this.linearWeight = new LinearImpl(sparseDim, 1);
        register_module("linear", linearWeight);
        linearWeight.to(new Device(device), false);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        // Get embeddings: (batch, num_fields, embed_dim)
        Tensor embeddings3D = embeddingLayer.forward3D(sparseFeats);
        long batchSize = embeddings3D.size(0);

        // 1) CIN: explicit high-order feature interactions
        Tensor cinOut = cin.forward(embeddings3D);  // (batch, 1) or (batch,)

        // 2) FM: 2nd-order feature interactions
        Tensor fmOut = fmLayer.forward(embeddings3D).squeeze(1);  // (batch,)

        // 3) Flatten 3D embeddings to 2D for DNN and linear
        Tensor flatEmbeddings = embeddings3D.view(batchSize, sparseDim);

        // 4) DNN: implicit high-order interactions
        Tensor mlpOut = mlp.forward(flatEmbeddings).squeeze(1);  // (batch,)

        // 5) Linear: 1st-order contributions
        Tensor linearOut = linearWeight.forward(flatEmbeddings).squeeze(1);  // (batch,)

        // Combine all: logit = linear + cin + fm + mlp
        // cinOut may be (batch, 1) — squeeze if needed to match
        Tensor cinScalar = cinOut.dim() > 1 ? cinOut.squeeze(1) : cinOut;
        return linearOut.add(cinScalar).add(fmOut).add(mlpOut);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
