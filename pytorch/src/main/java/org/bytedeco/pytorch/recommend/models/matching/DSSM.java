/*
 * Ported from torch-rechub-scala: torchrec/models/matching/DSSM.scala
 *
 * Deep Structured Semantic Model (DSSM) - Two Tower Architecture.
 * Reference: Microsoft
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DSSM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer userEmbedding;
    private final MLP userTower;
    private final EmbeddingLayer itemEmbedding;
    private final MLP itemTower;

    public DSSM(List<? extends Feature> userFeatures, List<? extends Feature> itemFeatures) {
        this(userFeatures, itemFeatures, 8, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public DSSM(List<? extends Feature> userFeatures, List<? extends Feature> itemFeatures,
                int embedDim, long[] towerDims, float dropout, String device) {
        super("DSSM");
        List<Feature> userList = new ArrayList<>(userFeatures);
        List<Feature> itemList = new ArrayList<>(itemFeatures);

        this.userEmbedding = new EmbeddingLayer(userList, embedDim, device);
        register_module("userEmbedding", userEmbedding);

        long userSparseDim = Features.calcSparseDim(userList);
        long userSeqDim = Features.calcSequenceDimFromFeatures(userList, "mean");
        long userTowerInputDim = userSparseDim + userSeqDim;
        this.userTower = new MLP(userTowerInputDim, towerDims, embedDim, "relu", dropout, false, device);
        register_module("userTower", userTower);

        this.itemEmbedding = new EmbeddingLayer(itemList, embedDim, device);
        register_module("itemEmbedding", itemEmbedding);

        long itemSparseDim = Features.calcSparseDim(itemList);
        long itemSeqDim = Features.calcSequenceDimFromFeatures(itemList, "mean");
        long itemTowerInputDim = itemSparseDim + itemSeqDim;
        this.itemTower = new MLP(itemTowerInputDim, towerDims, embedDim, "relu", dropout, false, device);
        register_module("itemTower", itemTower);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            userTower.to(dev, false);
            itemTower.to(dev, false);
        }
    }

    public Tensor forward(Map<String, Tensor> userFeats, Map<String, Tensor> itemFeats) {
        Tensor userEmb = userEmbedding.forward(userFeats);
        Tensor itemEmb = itemEmbedding.forward(itemFeats);

        Tensor userOut = userTower.forward(userEmb);
        Tensor itemOut = itemTower.forward(itemEmb);

        // Cosine similarity
        Tensor userNorm = userOut.pow(new Scalar(2)).sum(1).sqrt();
        Tensor itemNorm = itemOut.pow(new Scalar(2)).sum(1).sqrt();
        Tensor prodNorms = userNorm.mul(itemNorm);
        Tensor cosSim = userOut.mul(itemOut).sum(1).div(prodNorms.add(new Scalar(1e-8f)));
        return cosSim.unsqueeze(1);
    }

    public Tensor userTowerForward(Map<String, Tensor> userFeats) {
        Map<String, Tensor> sparseFeats = new LinkedHashMap<>();
        Map<String, Tensor> sequenceFeats = new LinkedHashMap<>();
        partitionFeats(userFeats, sparseFeats, sequenceFeats);
        Tensor userEmb = userEmbedding.forward(sparseFeats, sequenceFeats, false);
        Tensor squeezed = (userEmb.dim() == 2L && userEmb.size(1) == 1L)
                ? userEmb.squeeze(1) : userEmb;
        return userTower.forward(squeezed);
    }

    public Tensor itemTowerForward(Map<String, Tensor> itemFeats) {
        Map<String, Tensor> sparseFeats = new LinkedHashMap<>();
        Map<String, Tensor> sequenceFeats = new LinkedHashMap<>();
        partitionFeats(itemFeats, sparseFeats, sequenceFeats);
        Tensor itemEmb = itemEmbedding.forward(sparseFeats, sequenceFeats, false);
        Tensor squeezed = (itemEmb.dim() == 2L && itemEmb.size(1) == 1L)
                ? itemEmb.squeeze(1) : itemEmb;
        return itemTower.forward(squeezed);
    }

    /**
     * Partition features:
     * - 1D (batch,) -> sparse
     * - 2D (batch, 1) -> sparse; (batch, seqLen>1) -> sequence
     * - 3D (batch, 1, seqLen) -> sequence; else sparse
     */
    private static void partitionFeats(Map<String, Tensor> feats,
                                       Map<String, Tensor> sparseOut,
                                       Map<String, Tensor> seqOut) {
        for (Map.Entry<String, Tensor> e : feats.entrySet()) {
            Tensor t = e.getValue();
            boolean isSparse;
            switch ((int) t.dim()) {
                case 1:
                    isSparse = true;
                    break;
                case 2:
                    isSparse = t.size(1) == 1L;
                    break;
                case 3:
                    isSparse = t.size(1) != 1L;
                    break;
                default:
                    isSparse = false;
                    break;
            }
            if (isSparse) {
                sparseOut.put(e.getKey(), t);
            } else {
                Tensor processed = (t.dim() == 3L && t.size(1) == 1L) ? t.squeeze(1) : t;
                seqOut.put(e.getKey(), processed);
            }
        }
    }
}
