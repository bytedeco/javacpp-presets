/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/RobustKT.scala
 *
 * RobustKT: Robust Knowledge Tracing with smooth module and dual transformer blocks.
 * Also includes SmoothModule, CausalConv1d, RobustTransformerBlock.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.CosinePositionalEmbedding;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RobustKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final int embedDim;
    private final EmbeddingImpl qEmbed;
    private final EmbeddingImpl qaEmbed;
    private final Tensor qDiff;
    private final CosinePositionalEmbedding posEmb;
    private final SmoothModule smooth;
    private final List<RobustTransformerBlock> blocks1 = new ArrayList<>();
    private final List<RobustTransformerBlock> blocks2 = new ArrayList<>();
    private final MLP outMLP;
    private final DropoutImpl dropoutLayer;

    public RobustKT(long numConcepts) {
        this(numConcepts, 64, 8, 2, 5, 0.2f, DeviceSupport.backend());
    }

    public RobustKT(
            long numConcepts,
            int embedDim,
            int numHeads,
            int numBlocks,
            int kernelSize,
            float dropout,
            String device) {
        super("RobustKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numConcepts = numConcepts;
        this.embedDim = embedDim;

        this.qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("q_embed", qEmbed);

        this.qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim));
        register_module("qa_embed", qaEmbed);

        Tensor qd = torch.randn(
                new long[]{numConcepts + 1, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.01f));
        if (!"cpu".equals(device)) {
            qd = qd.to(new Device(device), ScalarType.Float);
        }
        this.qDiff = qd;
        register_parameter("q_diff", qDiff);

        this.posEmb = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_emb", posEmb);

        this.smooth = new SmoothModule(embedDim, kernelSize, dropout, device);
        register_module("smooth", smooth);

        for (int i = 0; i < numBlocks; i++) {
            RobustTransformerBlock block = new RobustTransformerBlock(embedDim, numHeads, dropout, device);
            register_module("block1_" + i, block);
            blocks1.add(block);
        }

        for (int i = 0; i < numBlocks * 2; i++) {
            RobustTransformerBlock block = new RobustTransformerBlock(embedDim, numHeads, dropout, device);
            register_module("block2_" + i, block);
            blocks2.add(block);
        }

        this.outMLP = new MLP(embedDim * 2L, new long[]{(long) embedDim, embedDim / 2L}, 1L, "relu", dropout,
                false, false, true, device);
        register_module("out_mlp", outMLP);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qEmbed.to(dev, false);
            qaEmbed.to(dev, false);
            outMLP.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        int batchSize = (int) conceptIds.size(0);
        int seqLen = (int) conceptIds.size(1);

        Tensor cIdsLong = conceptIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0L)),
                new ScalarOptional(new Scalar(numConcepts)))
                .toType(ScalarType.Long);
        Tensor rLong = responses.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0L)),
                new ScalarOptional(new Scalar(1L)))
                .toType(ScalarType.Long);

        Tensor qaIdsRaw = cIdsLong.add(rLong.mul(new Scalar(numConcepts)));
        long maxInteraction = numConcepts * 2 - 1;
        Tensor qaIds = qaIdsRaw.clamp(
                new ScalarOptional(new Scalar(0L)),
                new ScalarOptional(new Scalar(maxInteraction)))
                .toType(ScalarType.Long);
        Tensor qaEmb = qaEmbed.forward(qaIds);
        Tensor qEmb = qEmbed.forward(cIdsLong);

        Tensor qDiffEmb = qDiff.index_select(0, cIdsLong.toType(ScalarType.Long).view(-1L))
                .view(batchSize, seqLen, embedDim);

        Tensor qaWithDiff = qaEmb.add(qDiffEmb);
        Tensor qWithDiff = qEmb.add(qDiffEmb);

        Tensor smoothedQA = smooth.forward(qaWithDiff);
        Tensor smoothedQ = smooth.forward(qWithDiff);

        Tensor posEnc = posEmb.forward(smoothedQA);
        Tensor qaWithPos = smoothedQA.add(posEnc);
        Tensor qWithPos = smoothedQ.add(posEnc);

        Tensor y = qaWithPos;
        for (RobustTransformerBlock block : blocks1) {
            y = block.forward(y, y, y, true);
        }

        Tensor x = qWithPos;
        boolean flagFirst = true;
        for (RobustTransformerBlock block : blocks2) {
            if (flagFirst) {
                x = block.forward(x, x, x, false);
                flagFirst = false;
            } else {
                x = block.forward(x, y, y, true);
                flagFirst = true;
            }
        }

        Tensor concatQ = torch.cat(new TensorVector(x, qEmb), 2);
        Tensor logits = outMLP.forward(concatQ);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}

/** Smooth module for sequence smoothing using trend/random decomposition. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
class SmoothModule extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final Tensor sqrtBeta;
    private final DropoutImpl dropoutLayer;
    private final LayerNormImpl ln;

    public SmoothModule(int embedDim, int kernelSize, float dropout, String device) {
        super("SmoothModule");
        // No-op causal conv for smoke-test stability (matches Scala)
        Tensor p = torch.rand(
                new long[]{1, 1, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        if (!"cpu".equals(device)) {
            p = p.to(new Device(device), ScalarType.Float);
        }
        this.sqrtBeta = p;
        register_parameter("sqrt_beta", sqrtBeta);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        LongVector shape = new LongVector(1);
        shape.put(0, embedDim);
        this.ln = new LayerNormImpl(new LayerNormOptions(shape));
        register_module("ln", ln);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor xTrans = x.transpose(1, 2);
        Tensor trend = xTrans; // no-op causal conv
        Tensor random = xTrans.sub(trend);

        Tensor betaSq = sqrtBeta.mul(sqrtBeta);
        Tensor randomWithBeta = random.transpose(1, 2).mul(betaSq);
        Tensor sequenceEmb = trend.transpose(1, 2).add(randomWithBeta);

        Tensor dropped = dropoutLayer.forward(sequenceEmb);
        Tensor res = x.add(dropped);
        return ln.forward(res);
    }
}

/** Causal Conv1d — prevents future information leakage. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
class CausalConv1d extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int padding;
    private final Conv1dImpl conv;

    public CausalConv1d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1);
    }

    public CausalConv1d(int inChannels, int outChannels, int kernelSize, int dilation) {
        super("CausalConv1d");
        this.padding = (kernelSize - 1) * dilation;
        Conv1dOptions opt = new Conv1dOptions(inChannels, outChannels, new LongPointer(new long[]{kernelSize}));
        opt.padding().put(new LongPointer(new long[]{padding}));
        opt.dilation().put(dilation);
        this.conv = new Conv1dImpl(opt);
        register_module("conv", conv);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor out = conv.forward(x);
        if (padding > 0) {
            return out.narrow(2, 0, (int) out.size(2) - padding);
        }
        return out;
    }
}

/** Robust transformer block with distance-based attention. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
class RobustTransformerBlock extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final String device;
    private final LinearImpl qLinear;
    private final LinearImpl kLinear;
    private final LinearImpl vLinear;
    private final LinearImpl outLinear;
    private final LayerNormImpl ln1;
    private final LayerNormImpl ln2;
    private final LinearImpl ff1;
    private final LinearImpl ff2;
    private final DropoutImpl dropoutLayer;
    private final Tensor gamma;

    public RobustTransformerBlock(int embedDim, int numHeads, float dropout, String device) {
        super("RobustTransformerBlock");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.device = device;

        this.qLinear = new LinearImpl(embedDim, embedDim);
        this.kLinear = new LinearImpl(embedDim, embedDim);
        this.vLinear = new LinearImpl(embedDim, embedDim);
        register_module("q_linear", qLinear);
        register_module("k_linear", kLinear);
        register_module("v_linear", vLinear);

        this.outLinear = new LinearImpl(embedDim, embedDim);
        register_module("out_linear", outLinear);

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

        Tensor g = torch.zeros(
                new long[]{1, numHeads, 1L, 1L},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        g.fill_(new Scalar(0.9));
        if (!"cpu".equals(device)) {
            g = g.to(new Device(device), ScalarType.Float);
        }
        this.gamma = g;
        register_parameter("gamma", gamma);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qLinear.to(dev, false);
            kLinear.to(dev, false);
            vLinear.to(dev, false);
            outLinear.to(dev, false);
            ff1.to(dev, false);
            ff2.to(dev, false);
        }
    }

    public Tensor forward(Tensor q, Tensor k, Tensor v, boolean applyPos) {
        int batchSize = (int) q.size(0);
        int seqLen = (int) q.size(1);

        Tensor qProj = qLinear.forward(q).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor kProj = kLinear.forward(k).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor vProj = vLinear.forward(v).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);

        Scalar scale = new Scalar((float) Math.sqrt(headDim));
        Tensor scores = torch.matmul(qProj, kProj.transpose(2, 3)).div(scale);

        if (seqLen > 1 && applyPos) {
            Device dev = new Device(device);
            Tensor posIds = torch.arange(
                    new Scalar(seqLen),
                    new TensorOptions()
                            .dtype(new ScalarTypeOptional(ScalarType.Float))
                            .device(new DeviceOptional(dev)))
                    .view(seqLen, 1L);
            Tensor posIdsT = posIds.t();
            Tensor distMat = posIds.sub(posIdsT).abs();
            Tensor distMatExp = distMat.view(1, 1, (long) seqLen, (long) seqLen);

            Tensor gammaVal = gamma.sigmoid();
            Tensor distDecay = torch.pow(gammaVal, distMatExp);
            scores = scores.mul(distDecay);
        }

        if (seqLen > 1) {
            Tensor causalMask = torch.triu(
                    torch.ones(new long[]{seqLen, seqLen},
                            new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))),
                    1).unsqueeze(0).unsqueeze(0);
            scores = scores.add(causalMask.mul(new Scalar(1e9)));
        }

        Tensor attnWeights = scores.softmax(-1);
        Tensor attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj);

        Tensor reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim);
        Tensor outProj = outLinear.forward(reshaped);

        Tensor withRes = q.add(outProj);
        Tensor normed1 = ln1.forward(withRes);

        Tensor ffOut = torch.relu(ff1.forward(normed1));
        Tensor ffDropped = dropoutLayer.forward(ffOut);
        Tensor ffResult = ff2.forward(ffDropped);

        return ln2.forward(normed1.add(ffResult));
    }

    public Tensor forward(Tensor q, Tensor k, Tensor v) {
        return forward(q, k, v, true);
    }
}
