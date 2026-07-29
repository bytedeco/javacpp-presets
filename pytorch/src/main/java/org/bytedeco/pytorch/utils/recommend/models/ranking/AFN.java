/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/AFN.scala
 *
 * Adaptive Factorization Network (AFN).
 * Sparse Input → EmbeddingLayer → LNN → MLP → Output
 * Reference: AAAI 2020
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AFN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final int embedDim;
    private final int lnnDim;
    private final long lnnOutputDim;
    private final EmbeddingLayer embeddingLayer;
    private final Tensor lnnWeight;
    private final Tensor lnnBias;
    private final DropoutImpl dropoutLayer;
    private final MLP mlp;

    public AFN(List<? extends Feature> features) {
        this(features, 8, 8, new long[]{256L, 128L, 64L}, 0.2f, DeviceSupport.backend());
    }

    public AFN(List<? extends Feature> features, int embedDim, int lnnDim,
               long[] mlpDims, float dropout, String device) {
        super("AFN");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        int nf = 0;
        for (Feature f : features) {
            if (f instanceof SparseFeature) nf++;
        }
        if (nf < 2) {
            throw new IllegalArgumentException("AFN requires at least 2 sparse features");
        }
        this.numFields = nf;
        this.embedDim = embedDim;
        this.lnnDim = lnnDim;
        this.lnnOutputDim = (long) lnnDim * embedDim;

        this.embeddingLayer = new EmbeddingLayer(new ArrayList<>(features), embedDim, device);
        register_module("embedding", embeddingLayer);

        // LNN weight: (lnn_dim, num_fields)
        TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        Tensor w = torch.randn(new long[]{lnnDim, numFields}, opts)
                .mul(new Scalar((float) Math.sqrt(2.0 / numFields)));
        register_parameter("lnn_weight", w);
        this.lnnWeight = w;

        // LNN bias: (lnn_dim, embed_dim)
        Tensor b = torch.zeros(new long[]{lnnDim, embedDim}, opts);
        register_parameter("lnn_bias", b);
        this.lnnBias = b;

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout_lnn", dropoutLayer);

        this.mlp = new MLP(lnnOutputDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            lnnWeight.to(dev, ScalarType.Float);
            lnnBias.to(dev, ScalarType.Float);
            mlp.to(dev, false);
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        // Get embeddings: (batch, num_fields, embed_dim)
        Tensor embeddings = embeddingLayer.forward3D(sparseFeats);
        int batchSize = (int) embeddings.size(0);

        // LNN: log(1 + |x|) transformation
        Tensor absEmbeddings = embeddings.abs();
        Tensor logEmbeddings = torch.log1p(absEmbeddings);

        Tensor w = lnnWeight.to(embeddings.device(), ScalarType.Float);
        Tensor b = lnnBias.to(embeddings.device(), ScalarType.Float);

        // logEmbeddings: (B, F, E) → transpose to (B, E, F)
        Tensor logEmbeddingsT = logEmbeddings.transpose(1, 2);
        Tensor wT = w.t(); // (F, L)
        // preAct: (B, E, L)
        Tensor preAct = torch.matmul(logEmbeddingsT, wT);
        // preActT: (B, L, E)
        Tensor preActT = preAct.transpose(1, 2);

        // Add bias
        Tensor bBcast = b.unsqueeze(0).expand(batchSize, lnnDim, embedDim);
        Tensor lnnOut = preActT.add(bBcast);

        // expm1: exp(x) - 1
        lnnOut = torch.expm1(lnnOut);

        // ReLU activation
        lnnOut = lnnOut.relu();

        // Dropout
        lnnOut = dropoutLayer.forward(lnnOut);

        // Flatten: (batch, lnn_dim * embed_dim)
        Tensor lnnFlat = lnnOut.reshape(batchSize, lnnOutputDim);

        return mlp.forward(lnnFlat).squeeze(1);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
