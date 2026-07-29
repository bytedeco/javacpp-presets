/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/RKT.scala
 * (RelationAttentionLayer)
 *
 * Relation-aware multi-head attention with causal mask.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RelationAttentionLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final LinearImpl qLinear;
    private final LinearImpl kLinear;
    private final LinearImpl vLinear;
    private final LinearImpl outLinear;
    private final DropoutImpl dropoutLayer;
    private final LayerNormImpl ln;

    public RelationAttentionLayer(int embedDim, int numHeads) {
        this(embedDim, numHeads, 0.1f, DeviceSupport.backend());
    }

    public RelationAttentionLayer(int embedDim, int numHeads, float dropout, String device) {
        super("RelationAttentionLayer");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;

        this.qLinear = new LinearImpl(embedDim, embedDim);
        this.kLinear = new LinearImpl(embedDim, embedDim);
        this.vLinear = new LinearImpl(embedDim, embedDim);
        this.outLinear = new LinearImpl(embedDim, embedDim);
        this.dropoutLayer = new DropoutImpl(dropout);

        LongVector shape = new LongVector(1);
        shape.put(0, embedDim);
        this.ln = new LayerNormImpl(new LayerNormOptions(shape));

        register_module("q_linear", qLinear);
        register_module("k_linear", kLinear);
        register_module("v_linear", vLinear);
        register_module("out_linear", outLinear);
        register_module("ln", ln);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qLinear.to(dev, false);
            kLinear.to(dev, false);
            vLinear.to(dev, false);
            outLinear.to(dev, false);
        }
    }

    public Tensor forward(
            Tensor q, Tensor k, Tensor v,
            Tensor relations,
            Tensor l1, Tensor l2) {
        int batchSize = (int) q.size(0);
        int seqLen = (int) q.size(1);

        Tensor qProj = qLinear.forward(q).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor kProj = kLinear.forward(k).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor vProj = vLinear.forward(v).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);

        Scalar scale = new Scalar((float) Math.sqrt(headDim));
        Tensor scores = torch.matmul(qProj, kProj.transpose(2, 3)).div(scale);

        // l1/l2 combination weights are computed but standard attention used (matches Scala)
        // float l1Val = l1.mean().item().toFloat();
        // float l2Val = l2.mean().item().toFloat();

        if (seqLen > 1) {
            Tensor causalMask = torch.triu(
                    torch.ones(new long[]{seqLen, seqLen},
                            new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))),
                    1).unsqueeze(0).unsqueeze(0);
            Tensor maskedScores = scores.add(causalMask.mul(new Scalar(1e9)));

            Tensor attnWeights = maskedScores.softmax(-1);
            Tensor attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj);

            Tensor reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim);
            Tensor out = outLinear.forward(reshaped);
            return ln.forward(q.add(out));
        } else {
            Tensor attnWeights = scores.softmax(-1);
            Tensor attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj);

            Tensor reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim);
            Tensor out = outLinear.forward(reshaped);
            return ln.forward(q.add(out));
        }
    }
}
