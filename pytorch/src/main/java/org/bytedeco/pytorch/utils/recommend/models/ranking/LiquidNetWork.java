/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/LiquidNetWork.scala
 *
 * Liquid ODE network over sequence embeddings + sparse features → MLP.
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LiquidNetWork extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> features;
    private final List<SequenceFeature> sequenceFeatures;
    private final int embedDim;
    private final int numOdeSteps;
    private final long sparseDim;
    private final Device targetDevice;
    private final EmbeddingLayer sparseEmbedding;
    private final EmbeddingLayer seqEmbedding;
    private final LiquidCell liquidCell;
    private final LinearImpl inputProj;
    private final LinearImpl outputProj;
    private final MLP mlp;

    public LiquidNetWork(List<? extends Feature> features, List<SequenceFeature> sequenceFeatures) {
        this(features, sequenceFeatures, 8, 16, 3, new long[]{64L, 32L}, 0.2f, DeviceSupport.backend());
    }

    public LiquidNetWork(List<? extends Feature> features, List<SequenceFeature> sequenceFeatures,
                         int embedDim, int hiddenDim, int numOdeSteps, long[] mlpDims,
                         float dropout, String device) {
        super("LiquidNetWork");
        this.features = new ArrayList<>(features);
        this.sequenceFeatures = new ArrayList<>(sequenceFeatures);
        this.embedDim = embedDim;
        this.numOdeSteps = numOdeSteps;
        this.targetDevice = new Device(device);
        this.sparseDim = (long) this.features.size() * embedDim;

        this.sparseEmbedding = new EmbeddingLayer(this.features, embedDim, device);
        this.seqEmbedding = new EmbeddingLayer(new ArrayList<>(this.sequenceFeatures), embedDim, device);
        register_module("sparseEmbedding", sparseEmbedding);
        register_module("seqEmbedding", seqEmbedding);

        this.liquidCell = new LiquidCell(embedDim, hiddenDim, device);
        register_module("liquidCell", liquidCell);

        this.inputProj = new LinearImpl(embedDim, hiddenDim);
        this.outputProj = new LinearImpl(hiddenDim, embedDim);
        inputProj.to(targetDevice, false);
        outputProj.to(targetDevice, false);
        register_module("inputProj", inputProj);
        register_module("outputProj", outputProj);

        long mlpInputDim = sparseDim + embedDim;
        this.mlp = new MLP(mlpInputDim, mlpDims, 1L, "relu", dropout, false, false, true, device);
        register_module("mlp", mlp);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> sequenceFeats) {
        long batchSize;
        if (sparseFeats != null && !sparseFeats.isEmpty()) {
            batchSize = sparseFeats.values().iterator().next().size(0);
        } else if (sequenceFeats != null && !sequenceFeats.isEmpty()) {
            batchSize = sequenceFeats.values().iterator().next().size(0);
        } else {
            batchSize = 128L;
        }

        TensorOptions tensorOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(ScalarType.Float))
                .device(new DeviceOptional(targetDevice));

        // --- 1. Sparse feature Embedding ---
        Tensor sparseEmb;
        if (sparseFeats != null && !sparseFeats.isEmpty() && !features.isEmpty()) {
            Tensor emb = sparseEmbedding.forward(sparseFeats, Collections.emptyMap(), true);
            sparseEmb = emb.dim() == 1 ? emb.unsqueeze(0) : emb;
        } else {
            sparseEmb = torch.zeros(new long[]{batchSize, sparseDim}, tensorOpts);
        }

        // --- 2. Sequence feature Embedding ---
        Tensor seqEmbRaw;
        if (sequenceFeats != null && !sequenceFeats.isEmpty() && !sequenceFeatures.isEmpty()) {
            Tensor raw = seqEmbedding.forward(Collections.emptyMap(), sequenceFeats, false);
            seqEmbRaw = raw.dim() == 4 ? raw.squeeze(0) : raw;
        } else {
            seqEmbRaw = torch.zeros(new long[]{batchSize, 20L, embedDim}, tensorOpts);
        }

        // Force 3D if flattened to 2D [batch, seqLen * embedDim]
        Tensor seqEmb;
        if (seqEmbRaw.dim() == 2) {
            long slen = seqEmbRaw.size(1) / embedDim;
            seqEmb = seqEmbRaw.view(batchSize, slen, embedDim);
        } else {
            seqEmb = seqEmbRaw;
        }

        int seqLen = (int) seqEmb.size(1);

        // --- 3. ODE initial state ---
        Tensor firstStep = seqEmb.select(1, 0);
        Tensor hidden = inputProj.forward(firstStep);
        Scalar dt = new Scalar(1f / numOdeSteps);

        // --- 4. ODE integration (Euler) ---
        for (int step = 0; step < numOdeSteps; step++) {
            float t = (float) step / numOdeSteps;
            float f = t * (seqLen - 1);
            int i = Math.min((int) f, seqLen - 2);
            float a = f - i;

            Tensor x0 = seqEmb.select(1, i);
            Tensor x1 = seqEmb.select(1, i + 1);
            Tensor xt = x0.mul(new Scalar(1f - a)).add(x1.mul(new Scalar(a)));

            Tensor dh = liquidCell.forward(hidden, xt, t);
            hidden = hidden.add(dh.mul(dt));
        }

        // --- 5. Project back and concat ---
        Tensor seqOut = outputProj.forward(hidden);
        TensorVector vec = new TensorVector();
        vec.push_back(sparseEmb);
        vec.push_back(seqOut);
        Tensor combined = torch.cat(vec, 1);

        // --- 6. MLP output ---
        return mlp.forward(combined);
    }
}
