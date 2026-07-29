/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/UKT.scala
 *
 * UKT: Uncertainty-aware Knowledge Tracing with stochastic mean/cov embeddings
 * and UncertaintyTransformerBlock.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers.CosinePositionalEmbedding;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class UKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final int embedDim;
    private final String device;
    private final EmbeddingImpl meanQEmbed;
    private final Tensor covQEmbed;
    private final EmbeddingImpl meanQAEmbed;
    private final Tensor covQAEmbed;
    private final Tensor qDiff;
    private final CosinePositionalEmbedding posMeanEmbed;
    private final CosinePositionalEmbedding posCovEmbed;
    private final List<UncertaintyTransformerBlock> blocks = new ArrayList<>();
    private final MLP outMLP;
    private final DropoutImpl dropoutLayer;

    public UKT(long numConcepts) {
        this(numConcepts, 64, 8, 2, 0.2f, DeviceSupport.backend());
    }

    public UKT(
            long numConcepts,
            int embedDim,
            int numHeads,
            int numBlocks,
            float dropout,
            String device) {
        super("UKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numConcepts = numConcepts;
        this.embedDim = embedDim;
        this.device = device;

        this.meanQEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("mean_q_embed", meanQEmbed);

        Tensor covQ = torch.randn(
                new long[]{numConcepts + 1, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.01f));
        this.covQEmbed = covQ;
        register_parameter("cov_q_embed", covQEmbed);

        this.meanQAEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim));
        register_module("mean_qa_embed", meanQAEmbed);

        Tensor covQA = torch.randn(
                new long[]{numConcepts * 2, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.01f));
        this.covQAEmbed = covQA;
        register_parameter("cov_qa_embed", covQAEmbed);

        Tensor qd = torch.randn(
                new long[]{numConcepts + 1, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.01f));
        this.qDiff = qd;
        register_parameter("q_diff", qDiff);

        this.posMeanEmbed = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_mean_embed", posMeanEmbed);

        this.posCovEmbed = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_cov_embed", posCovEmbed);

        for (int i = 0; i < numBlocks; i++) {
            UncertaintyTransformerBlock block = new UncertaintyTransformerBlock(embedDim, numHeads, dropout, device);
            register_module("block_" + i, block);
            blocks.add(block);
        }

        this.outMLP = new MLP(embedDim * 4L, new long[]{embedDim * 2L, (long) embedDim}, 1L, "relu", dropout,
                false, false, true, device);
        register_module("out_mlp", outMLP);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            meanQEmbed.to(dev, false);
            meanQAEmbed.to(dev, false);
            outMLP.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        int batchSize = (int) conceptIds.size(0);
        int seqLen = (int) conceptIds.size(1);
        Device dev = new Device(device);

        Tensor cIdsLong = conceptIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numConcepts)));
        Tensor rLong = responses.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(1)));

        Tensor cIdsLongDev = cIdsLong.to(dev, ScalarType.Long);
        Tensor rLongDev = rLong.to(dev, ScalarType.Long);

        Tensor qMeanEmb = meanQEmbed.forward(cIdsLongDev);
        Tensor qaIndex = cIdsLongDev.add(rLongDev.mul(new Scalar((double) numConcepts)))
                .toType(ScalarType.Long).to(dev, ScalarType.Long);
        Tensor qaMeanEmb = meanQAEmbed.forward(qaIndex);

        Tensor qCovEmb = covQEmbed.index_select(0, cIdsLongDev.view(-1L)).view(batchSize, seqLen, embedDim);
        Tensor qaIndexFlat = qaIndex.view(-1L);
        Tensor qaCovEmb = covQAEmbed.index_select(0, qaIndexFlat).view(batchSize, seqLen, embedDim);

        Tensor qDiffEmb = qDiff.index_select(0, cIdsLongDev.view(-1L)).view(batchSize, seqLen, embedDim);

        Tensor qMeanPos = posMeanEmbed.forward(qMeanEmb);
        Tensor qCovPos = posCovEmbed.forward(qCovEmb);

        Tensor qMeanWithPos = qMeanEmb.add(qMeanPos);
        Tensor qCovWithPos = qCovEmb.add(qCovPos).add(new Scalar(1.0));

        Tensor qaMeanWithPos = qaMeanEmb.add(qMeanPos);
        Tensor qaCovWithPos = qaCovEmb.add(qCovPos).add(new Scalar(1.0));

        Tensor qMeanAdjusted = qMeanWithPos.add(qDiffEmb);
        Tensor qCovAdjusted = qCovWithPos.add(qDiffEmb);

        Tensor meanOut = qMeanAdjusted;
        Tensor covOut = qCovAdjusted;
        for (UncertaintyTransformerBlock block : blocks) {
            T_TensorTensor_T result = block.forwardPair(meanOut, covOut, qaMeanWithPos, qaCovWithPos);
            meanOut = result.get0();
            covOut = result.get1();
        }

        Tensor covActivated = torch.elu(covOut).add(new Scalar(1.0));
        Tensor concatFeatures = torch.cat(new TensorVector(meanOut, covActivated, qMeanEmb, qCovEmb), 2);
        Tensor logits = outMLP.forward(concatFeatures);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}

/** Transformer block with uncertainty (mean and covariance) attention. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
class UncertaintyTransformerBlock extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int headDim;
    private final LinearImpl meanQ;
    private final LinearImpl meanK;
    private final LinearImpl meanV;
    private final LinearImpl covQ;
    private final LinearImpl covK;
    private final LinearImpl covV;
    private final LinearImpl meanOutProj;
    private final LinearImpl covOutProj;
    private final LayerNormImpl ln1;
    private final LayerNormImpl ln2;
    private final LinearImpl ff1;
    private final LinearImpl ff2;
    private final DropoutImpl dropoutLayer;

    public UncertaintyTransformerBlock(int embedDim, int numHeads, float dropout, String device) {
        super("UncertaintyTransformerBlock");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.headDim = embedDim / numHeads;

        this.meanQ = new LinearImpl(embedDim, embedDim);
        this.meanK = new LinearImpl(embedDim, embedDim);
        this.meanV = new LinearImpl(embedDim, embedDim);
        register_module("mean_q", meanQ);
        register_module("mean_k", meanK);
        register_module("mean_v", meanV);

        this.covQ = new LinearImpl(embedDim, embedDim);
        this.covK = new LinearImpl(embedDim, embedDim);
        this.covV = new LinearImpl(embedDim, embedDim);
        register_module("cov_q", covQ);
        register_module("cov_k", covK);
        register_module("cov_v", covV);

        this.meanOutProj = new LinearImpl(embedDim, embedDim);
        this.covOutProj = new LinearImpl(embedDim, embedDim);
        register_module("mean_out_proj", meanOutProj);
        register_module("cov_out_proj", covOutProj);

        LongVector shape = new LongVector(1);
        shape.put(0, embedDim);
        this.ln1 = new LayerNormImpl(new LayerNormOptions(shape));
        this.ln2 = new LayerNormImpl(new LayerNormOptions(shape));
        register_module("ln1", ln1);
        register_module("ln2", ln2);

        this.ff1 = new LinearImpl(embedDim, embedDim * 4L);
        this.ff2 = new LinearImpl(embedDim * 4L, embedDim);
        register_module("ff1", ff1);
        register_module("ff2", ff2);

        this.dropoutLayer = new DropoutImpl(dropout);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            meanQ.to(dev, false);
            meanK.to(dev, false);
            meanV.to(dev, false);
            covQ.to(dev, false);
            covK.to(dev, false);
            covV.to(dev, false);
            meanOutProj.to(dev, false);
            covOutProj.to(dev, false);
            ff1.to(dev, false);
            ff2.to(dev, false);
        }
    }

    public T_TensorTensor_T forwardPair(Tensor qMean, Tensor qCov, Tensor vMean, Tensor vCov) {
        int seqLen = (int) qMean.size(1);

        Tensor qM = meanQ.forward(qMean);
        Tensor kM = meanK.forward(vMean);
        Tensor vM = meanV.forward(vMean);

        Scalar scale = new Scalar((float) Math.sqrt(headDim));
        Tensor scores = torch.matmul(qM, kM.transpose(1, 2)).div(scale);

        if (seqLen > 1) {
            Tensor causalMask = torch.triu(
                    torch.ones(new long[]{seqLen, seqLen},
                            new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))),
                    1).unsqueeze(0);
            scores = scores.add(causalMask.mul(new Scalar(1e9)));
        }

        Tensor attnWeights = scores.softmax(-1);
        Tensor meanAttn = torch.matmul(dropoutLayer.forward(attnWeights), vM);
        Tensor meanProjected = meanOutProj.forward(meanAttn);

        Tensor meanWithRes = qMean.add(meanProjected);
        Tensor normedMean = ln1.forward(meanWithRes);

        Tensor ffOut = torch.relu(ff1.forward(normedMean));
        Tensor ffDropped = dropoutLayer.forward(ffOut);
        Tensor ffResult = ff2.forward(ffDropped);
        Tensor meanFinal = ln2.forward(normedMean.add(ffResult));

        Tensor covProjected = covOutProj.forward(qCov);
        Tensor covWithRes = qCov.add(covProjected);

        return new T_TensorTensor_T(meanFinal, covWithRes);
    }
}
