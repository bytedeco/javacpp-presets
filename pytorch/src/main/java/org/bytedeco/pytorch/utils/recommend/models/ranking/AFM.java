/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/AFM.scala
 *
 * Attentional Factorization Machine (AFM). Reference: IJCAI 2017.
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.FMInteraction;
import org.bytedeco.pytorch.utils.recommend.basic.layers.LR;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AFM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final int embedDim;
    private final LR linear;
    private final FMInteraction fm;
    private final EmbeddingLayer embeddingLayer;
    private final LinearImpl attentionLiner;
    private final Tensor h;
    private final Tensor p;
    private final DropoutImpl dropoutLayer;

    public AFM(List<SparseFeature> features) {
        this(features, 8, 64, 0.2f, DeviceSupport.backend());
    }

    public AFM(List<SparseFeature> features, int embedDim, int attentionDim,
               float dropout, String device) {
        super("AFM");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        this.numFields = features.size();
        if (numFields < 2) {
            throw new IllegalArgumentException("AFM requires at least 2 sparse features for interaction");
        }
        this.embedDim = embedDim;

        long fmDims = 0L;
        for (SparseFeature f : features) {
            fmDims += f.embedDim();
        }
        this.linear = new LR(fmDims, false, device);
        register_module("linear", linear);

        this.fm = new FMInteraction(embedDim);
        register_module("fm", fm);

        this.embeddingLayer = new EmbeddingLayer(new ArrayList<>(features), embedDim, device);
        register_module("embedding", embeddingLayer);

        this.attentionLiner = new LinearImpl(embedDim, attentionDim);
        register_module("attention_liner", attentionLiner);

        // h: (attentionDim, 1) Xavier-like uniform
        float stdH = (float) Math.sqrt(6.0 / (attentionDim + 1));
        Random random = new Random(42);
        float[] arrH = new float[attentionDim];
        for (int i = 0; i < attentionDim; i++) {
            arrH[i] = (random.nextFloat() * 2 - 1) * stdH;
        }
        Tensor hT = torch.tensor(arrH).view(attentionDim, 1L).toType(ScalarType.Float);
        register_parameter("h", hT);
        this.h = hT;

        // p: (embed_dim, 1)
        float stdP = (float) Math.sqrt(6.0 / (embedDim + 1));
        random = new Random(42);
        float[] arrP = new float[embedDim];
        for (int i = 0; i < embedDim; i++) {
            arrP[i] = (random.nextFloat() * 2 - 1) * stdP;
        }
        Tensor pT = torch.tensor(arrP).view(embedDim, 1L).toType(ScalarType.Float);
        register_parameter("p", pT);
        this.p = pT;

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        if (device != null && !"cpu".equals(device)) {
            this.to(new Device(device), false);
        }
    }

    private Tensor attention(Tensor yFm) {
        Tensor yAtt = attentionLiner.forward(yFm);
        Tensor yRelu = torch.relu(yAtt);
        Tensor yMatmul = torch.matmul(yRelu, h.to(yFm.device(), ScalarType.Float));
        return torch.softmax(yMatmul, 1);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings = embeddingLayer.forward3D(sparseFeats, Collections.emptyMap());
        int batchSize = (int) embeddings.size(0);

        Tensor embeddingsFlat = embeddings.view(batchSize, (long) numFields * embedDim);
        Tensor yLinear = linear.forward(embeddingsFlat);

        Tensor yFm = fm.forward(embeddings);
        Tensor atts = attention(yFm);
        Tensor attsDrop = dropoutLayer.forward(atts);

        Tensor weighted = attsDrop.mul(yFm);
        Tensor outs = torch.matmul(weighted, p.to(yFm.device(), ScalarType.Float));

        return yLinear.add(outs);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
