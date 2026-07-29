/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DeepFFM.scala
 *
 * Deep Field-weighted Factorization Machine (DeepFFM).
 * Includes ranking-local FFM and FatDeepFFM variant.
 * Reference: Alibaba, IJCAI 2018
 *
 * Note: ranking.DeepFFM.FFM differs from basic.layers.FFM (pairwise field products).
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Features;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Ranking-local Field-weighted FM (from DeepFFM.scala).
 * Formula mirrors classic FM 2nd-order on 3D embeddings → (batch, 1).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DeepFFMFieldFM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int fieldNum;

    public DeepFFMFieldFM(int embedDim, int fieldNum) {
        this(embedDim, fieldNum, DeviceSupport.backend());
    }

    public DeepFFMFieldFM(int embedDim, int fieldNum, String device) {
        super("FFM");
        this.embedDim = embedDim;
        this.fieldNum = fieldNum;
    }

    @Override
    public Tensor forward(Tensor embeddings) {
        // embeddings: (batch, num_fields, embed_dim)
        Tensor firstOrder = embeddings.sum(1); // (batch, embed_dim)

        Scalar twoScalar = new Scalar(2.0f);
        Tensor squaredSum = torch.pow(embeddings, twoScalar).sum(1);
        Tensor sumSquared = torch.pow(embeddings.sum(1), twoScalar);

        Scalar halfScalar = new Scalar(0.5f);
        Tensor interactions = sumSquared.sub(squaredSum).mul(halfScalar);

        return firstOrder.add(interactions).sum(1).unsqueeze(1);
    }
}
