/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/DIMKT.scala
 *
 * DIMKT: Data-aware Inductive Moment Knowledge Tracing.
 * Three concept views → cross-attention fusion → LSTM moment → MLP.
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
import org.bytedeco.pytorch.T_TensorT_TensorTensor_T_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LSTMImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.LSTMOptions;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers.MultiHeadAttention;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DIMKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final int embedDim;
    private final EmbeddingImpl conceptEmb1;
    private final EmbeddingImpl conceptEmb2;
    private final EmbeddingImpl conceptEmb3;
    private final EmbeddingImpl responseEmb;
    private final Tensor conceptPrior1;
    private final Tensor conceptPrior2;
    private final MultiHeadAttention crossAttn12;
    private final MultiHeadAttention crossAttn13;
    private final LSTMImpl momentLSTM;
    private final LinearImpl outputProj;
    private final MLP outMLP;
    private final DropoutImpl dropoutLayer;
    private final LayerNormImpl ln1;
    private final LayerNormImpl ln2;

    public DIMKT(long numConcepts) {
        this(numConcepts, 64, 8, 2, 64, 0.2f, DeviceSupport.backend());
    }

    public DIMKT(
            long numConcepts,
            int embedDim,
            int numHeads,
            int numBlocks,
            int hiddenDim,
            float dropout,
            String device) {
        super("DIMKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numConcepts = numConcepts;
        this.embedDim = embedDim;

        this.conceptEmb1 = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("concept_emb1", conceptEmb1);
        this.conceptEmb2 = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("concept_emb2", conceptEmb2);
        this.conceptEmb3 = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("concept_emb3", conceptEmb3);

        this.responseEmb = new EmbeddingImpl(new EmbeddingOptions(2 + 1, embedDim));
        register_module("response_emb", responseEmb);

        Tensor p1 = torch.randn(
                new long[]{numConcepts + 1, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.01f));
        this.conceptPrior1 = p1;
        register_parameter("concept_prior1", conceptPrior1);

        Tensor p2 = torch.randn(
                new long[]{numConcepts + 1, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.01f));
        this.conceptPrior2 = p2;
        register_parameter("concept_prior2", conceptPrior2);

        this.crossAttn12 = new MultiHeadAttention(embedDim, numHeads, dropout, device);
        register_module("cross_attn12", crossAttn12);
        this.crossAttn13 = new MultiHeadAttention(embedDim, numHeads, dropout, device);
        register_module("cross_attn13", crossAttn13);

        LSTMOptions lstmOptions = new LSTMOptions(embedDim, hiddenDim);
        lstmOptions.batch_first().put(true);
        lstmOptions.dropout().put((double) dropout);
        this.momentLSTM = new LSTMImpl(lstmOptions);
        register_module("moment_lstm", momentLSTM);

        this.outputProj = new LinearImpl(hiddenDim, embedDim);
        register_module("output_proj", outputProj);

        this.outMLP = new MLP(embedDim, new long[]{(long) embedDim}, 1L, "relu", dropout,
                false, false, true, device);
        register_module("out_mlp", outMLP);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        this.ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
        this.ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(hiddenDim)));
        register_module("ln1", ln1);
        register_module("ln2", ln2);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            conceptEmb1.to(dev, false);
            conceptEmb2.to(dev, false);
            conceptEmb3.to(dev, false);
            responseEmb.to(dev, false);
            outputProj.to(dev, false);
            outMLP.to(dev, false);
        }
    }

    private static LongVector layerNormShape(int d) {
        LongVector v = new LongVector(1);
        v.put(0, d);
        return v;
    }

    public Tensor forward(
            Tensor conceptIds1,
            Tensor conceptIds2,
            Tensor conceptIds3,
            Tensor responses) {
        int batchSize = (int) conceptIds1.size(0);
        int seqLen = (int) conceptIds1.size(1);

        Tensor c1Clamped = conceptIds1.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numConcepts)))
                .toType(ScalarType.Long);
        Tensor c2Clamped = conceptIds2.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numConcepts)))
                .toType(ScalarType.Long);
        Tensor c3Clamped = conceptIds3.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numConcepts)))
                .toType(ScalarType.Long);
        Tensor rClamped = responses.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(1)))
                .toType(ScalarType.Long);

        Tensor emb1 = conceptEmb1.forward(c1Clamped);
        Tensor emb2 = conceptEmb2.forward(c2Clamped);
        Tensor emb3 = conceptEmb3.forward(c3Clamped);
        Tensor resEmb = responseEmb.forward(rClamped);

        Tensor prior1 = conceptPrior1.index_select(0, c1Clamped.toType(ScalarType.Long).view(-1L))
                .view(batchSize, seqLen, embedDim);
        Tensor prior2 = conceptPrior2.index_select(0, c2Clamped.toType(ScalarType.Long).view(-1L))
                .view(batchSize, seqLen, embedDim);

        Tensor fused1 = emb1.add(prior1).add(resEmb);
        Tensor fused2 = emb2.add(prior1).add(resEmb);
        Tensor fused3 = emb3.add(prior2).add(resEmb);

        Tensor attn12 = crossAttn12.forward(fused1, fused2, fused2);
        Tensor combined12 = ln1.forward(fused1.add(dropoutLayer.forward(attn12)));

        Tensor attn13 = crossAttn13.forward(combined12, fused3, fused3);
        Tensor combined = ln1.forward(combined12.add(dropoutLayer.forward(attn13)));

        Tensor enriched = combined.add(fused3);

        T_TensorT_TensorTensor_T_T lstmRet = momentLSTM.forwardT_TensorT_TensorTensor_T_T(enriched);
        Tensor lstmOut = lstmRet.get0();

        Tensor projected = outputProj.forward(dropoutLayer.forward(lstmOut));
        // Note: Scala uses ln2 on projected.add(lstmOut) but dims may differ if
        // hiddenDim != embedDim; mirror source as written.
        Tensor residualOut = ln2.forward(projected.add(lstmOut));

        Tensor logits = outMLP.forward(residualOut);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(
            Tensor conceptIds1,
            Tensor conceptIds2,
            Tensor conceptIds3,
            Tensor responses) {
        return forward(conceptIds1, conceptIds2, conceptIds3, responses);
    }
}
