/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/RKT.scala
 *
 * RKT: Relation-aware Knowledge Tracing.
 * Embedding + Relation Matrix + Correlation Attention → LSTM → Prediction.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.T_TensorT_TensorTensor_T_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LSTMImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.LSTMOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers.CosinePositionalEmbedding;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final EmbeddingImpl conceptEmb;
    private final EmbeddingImpl responseEmb;
    private final Tensor relationMatrix;
    private final Tensor l1Weight;
    private final Tensor l2Weight;
    private final CosinePositionalEmbedding posEmb;
    private final LinearImpl inputProj;
    private final RelationAttentionLayer selfAttn;
    private final LSTMImpl lstm;
    private final DropoutImpl dropoutLayer;
    private final LinearImpl outputLayer;

    public RKT(long numConcepts) {
        this(numConcepts, 64, 4, 1, 0.2f, DeviceSupport.backend());
    }

    public RKT(
            long numConcepts,
            int embedDim,
            int numHeads,
            int numLayers,
            float dropout,
            String device) {
        super("RKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numConcepts = numConcepts;

        this.conceptEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("concept_emb", conceptEmb);

        this.responseEmb = new EmbeddingImpl(new EmbeddingOptions(2 + 1, embedDim));
        register_module("response_emb", responseEmb);

        long size = numConcepts + 1;
        Tensor relInit = torch.randn(
                new long[]{size, size},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.01f));
        this.relationMatrix = relInit;
        register_parameter("relation_matrix", relationMatrix);

        Tensor l1 = torch.rand(
                new long[]{1L},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        this.l1Weight = l1;
        register_parameter("l1_weight", l1Weight);

        Tensor l2 = torch.rand(
                new long[]{1L},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        this.l2Weight = l2;
        register_parameter("l2_weight", l2Weight);

        this.posEmb = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_emb", posEmb);

        this.inputProj = new LinearImpl(embedDim * 2L, embedDim);
        register_module("input_proj", inputProj);

        this.selfAttn = new RelationAttentionLayer(embedDim, numHeads, dropout, device);
        register_module("self_attn", selfAttn);

        LSTMOptions lstmOptions = new LSTMOptions(embedDim, embedDim);
        lstmOptions.num_layers().put(numLayers);
        lstmOptions.dropout().put((double) dropout);
        lstmOptions.batch_first().put(true);
        this.lstm = new LSTMImpl(lstmOptions);
        register_module("lstm", lstm);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        this.outputLayer = new LinearImpl(embedDim, 1);
        register_module("output", outputLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            conceptEmb.to(dev, false);
            responseEmb.to(dev, false);
            lstm.to(dev, false);
            outputLayer.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        int batchSize = (int) conceptIds.size(0);
        int seqLen = (int) conceptIds.size(1);

        Tensor cIdsLong = conceptIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numConcepts)));
        Tensor rLong = responses.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(1)));

        Tensor cEmb = conceptEmb.forward(cIdsLong.toType(ScalarType.Long));
        Tensor rEmb = responseEmb.forward(rLong.toType(ScalarType.Long));

        Tensor rExp = rLong.unsqueeze(2).toType(ScalarType.Float);
        Tensor inputEmb = torch.cat(new TensorVector(
                cEmb.mul(rExp),
                cEmb.mul(new Scalar(1.0)).sub(rExp)), 2);

        Tensor projected = inputProj.forward(inputEmb);
        Tensor posEnc = posEmb.forward(projected);
        Tensor withPos = projected.add(posEnc);

        Tensor relations = getRelationMatrix(cIdsLong, cIdsLong);
        Tensor attnOut = selfAttn.forward(withPos, withPos, withPos, relations, l1Weight, l2Weight);

        T_TensorT_TensorTensor_T_T lstmRet = lstm.forwardT_TensorT_TensorTensor_T_T(attnOut);
        Tensor lstmOut = lstmRet.get0();
        Tensor dropped = dropoutLayer.forward(lstmOut);
        Tensor logits = outputLayer.forward(dropped);
        return logits.sigmoid().squeeze(2);
    }

    /** Get relation matrix for concept pairs (zeros placeholder, matches Scala). */
    private Tensor getRelationMatrix(Tensor conceptIds1, Tensor conceptIds2) {
        int batchSize = (int) conceptIds1.size(0);
        int seqLen = (int) conceptIds1.size(1);
        return torch.zeros(
                new long[]{batchSize, seqLen, seqLen},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
