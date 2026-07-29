/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/HoFM.scala
 *
 * High-Order Factorization Machine (HoFM).
 * Sparse Input → EmbeddingLayer → [AnovaKernel order>=3] + FM (order 2) + MLP → Output
 * Reference: Rendle, 2010 — High-order extension
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.AnovaKernel;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.FMInteraction;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class HoFM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int order;
    private final long mlpInputDim;
    private final EmbeddingLayer embeddingLayer;
    private final FMInteraction fm;
    private final AnovaKernel anovaKernel; // non-null if order >= 3
    private final MLP mlp;

    public HoFM(List<? extends Feature> features) {
        this(features, 8, 3, new long[]{128L, 64L}, 0.2f, DeviceSupport.backend());
    }

    public HoFM(List<? extends Feature> features, int embedDim, int order,
                long[] mlpDims, float dropout, String device) {
        super("HoFM");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (order < 2) {
            throw new IllegalArgumentException("order must be >= 2, got " + order);
        }
        if (embedDim <= 0) {
            throw new IllegalArgumentException("embedDim must be positive, got " + embedDim);
        }
        this.order = order;

        int numFields = 0;
        for (Feature f : features) {
            if (f instanceof SparseFeature) numFields++;
        }
        if (numFields < order) {
            throw new IllegalArgumentException(
                    "numFields (" + numFields + ") must be >= order (" + order + ")");
        }

        this.embeddingLayer = new EmbeddingLayer(new ArrayList<>(features), embedDim, device);
        register_module("embedding", embeddingLayer);

        this.fm = new FMInteraction(embedDim);
        register_module("fm", fm);

        if (order >= 3) {
            this.anovaKernel = new AnovaKernel(order, embedDim, false, device);
            register_module("anova_kernel", anovaKernel);
        } else {
            this.anovaKernel = null;
        }

        // MLP input: embedDim * (order - 1) if order >= 3, else embedDim
        this.mlpInputDim = order >= 3 ? (long) embedDim * (order - 1) : embedDim;

        this.mlp = new MLP(mlpInputDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            mlp.to(dev, false);
            fm.to(dev, false);
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings = embeddingLayer.forward3D(sparseFeats);
        int batchSize = (int) embeddings.size(0);

        List<Tensor> interactionOutputs = new ArrayList<>();

        if (order >= 2) {
            interactionOutputs.add(fm.forward(embeddings)); // (batch, embed_dim)
        }

        if (order >= 3 && anovaKernel != null) {
            interactionOutputs.add(anovaKernel.forward(embeddings)); // (batch, embed_dim)
        }

        Tensor combined;
        if (interactionOutputs.size() == 1) {
            combined = interactionOutputs.get(0);
        } else {
            var targetDev = interactionOutputs.get(0).device();
            TensorVector tensorVec = new TensorVector();
            for (Tensor t : interactionOutputs) {
                Tensor onDev = t.device().equals(targetDev) ? t : t.to(targetDev, t.dtype());
                tensorVec.push_back(onDev);
            }
            combined = torch.cat(tensorVec, 1L);
        }

        Tensor mlpInput;
        if (combined.dim() == 3) {
            mlpInput = combined.view(batchSize, -1);
        } else {
            mlpInput = combined;
        }

        return mlp.forward(mlpInput).squeeze(1);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
