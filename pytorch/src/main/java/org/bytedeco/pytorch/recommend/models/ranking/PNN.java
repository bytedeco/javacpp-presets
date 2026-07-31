/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/PNN.scala
 *
 * Product-based Neural Network (PNN). Reference: Song et al., 2016.
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.InnerProductNetwork;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.basic.layers.OuterProductNetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class PNN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final String productType;
    private final int numFields;
    private final int embedDim;
    private final int embedFlatDim;
    private final int productDim;
    private final int mlpInputDim;
    private final EmbeddingLayer embeddingLayer;
    private final OuterProductNetwork outerNet; // null if inner
    private final MLP mlp;

    public PNN(List<? extends Feature> features) {
        this(features, 8, new long[]{256L, 128L, 64L}, "inner", 0.2f, DeviceSupport.backend());
    }

    public PNN(List<? extends Feature> features, int embedDim, long[] mlpDims,
               String productType, float dropout, String device) {
        super("PNN");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (!"inner".equals(productType) && !"outer".equals(productType)) {
            throw new IllegalArgumentException(
                    "productType must be 'inner' or 'outer', got " + productType);
        }
        this.productType = productType;
        this.embedDim = embedDim;

        int nf = 0;
        for (Feature f : features) {
            if (f instanceof SparseFeature) nf++;
        }
        if (nf < 2) {
            throw new IllegalArgumentException("PNN requires at least 2 sparse features");
        }
        this.numFields = nf;
        int numPairs = (numFields * (numFields - 1)) / 2;
        this.embedFlatDim = numFields * embedDim;
        this.productDim = "inner".equals(productType) ? numPairs : numPairs * embedDim;
        this.mlpInputDim = embedFlatDim + productDim;

        this.embeddingLayer = new EmbeddingLayer(new ArrayList<>(features), embedDim, device);
        register_module("embedding", embeddingLayer);

        if ("outer".equals(productType)) {
            this.outerNet = new OuterProductNetwork(numFields, embedDim, "vec", device);
            register_module("opn", outerNet);
        } else {
            this.outerNet = null;
        }

        this.mlp = new MLP(mlpInputDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);

        if (device != null && !"cpu".equals(device)) {
            mlp.to(new Device(device), false);
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings = embeddingLayer.forward3D(sparseFeats);
        long batch = embeddings.size(0);
        Tensor flatEmbed = embeddingLayer.forward(sparseFeats);

        Tensor productFeatures;
        if ("inner".equals(productType)) {
            productFeatures = new InnerProductNetwork().forward(embeddings);
        } else {
            if (outerNet == null) {
                throw new IllegalStateException("outerNet not initialized");
            }
            productFeatures = outerNet.forward(embeddings);
        }

        var targetDev = flatEmbed.device();
        TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        Tensor mlpInputTmp = torch.zeros(new long[]{batch, mlpInputDim}, opts)
                .to(targetDev, ScalarType.Float);

        Tensor left = flatEmbed.view(batch, embedFlatDim);
        mlpInputTmp.narrow(1L, 0L, embedFlatDim).copy_(left);

        Tensor prodOnDev;
        try {
            prodOnDev = productFeatures.to(targetDev, productFeatures.dtype());
        } catch (Throwable t) {
            long d0 = productFeatures.size(0);
            long d1 = productFeatures.dim() > 1L ? productFeatures.size(1) : 1L;
            Tensor tmp = torch.zeros(new long[]{d0, d1}, opts).to(targetDev, ScalarType.Float);
            tmp.copy_(productFeatures);
            prodOnDev = tmp;
        }
        Tensor right = prodOnDev.view(batch, productDim);
        mlpInputTmp.narrow(1L, embedFlatDim, productDim).copy_(right);

        return mlp.forward(mlpInputTmp).squeeze(1);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
