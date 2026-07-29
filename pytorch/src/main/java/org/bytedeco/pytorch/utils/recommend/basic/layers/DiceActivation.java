/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/DiceActivation.scala
 *
 * Dice activation function. Reference: Alibaba DIN paper, KDD 2018.
 * Formula: output = p * x + (1 - p) * alpha * x
 * where p = sigmoid(beta * bn(x)), and alpha, beta are learnable.
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;

/**
 * Dice activation function.
 * Input: (batch, embed) or (batch, seq, embed). Output: same shape as input.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DiceActivation extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedSize;
    private final float eps;
    private final BatchNorm1dImpl bn;
    private final Tensor alpha;
    private final Tensor beta;

    public DiceActivation(int embedSize) {
        this(embedSize, 1e-8f);
    }

    public DiceActivation(int embedSize, float eps) {
        super("DiceActivation");
        this.embedSize = embedSize;
        this.eps = eps;
        this.bn = new BatchNorm1dImpl(new BatchNormOptions(embedSize));
        register_module("bn", bn);

        // Learnable alpha and beta per dimension
        this.alpha = torch.zeros(new long[]{embedSize},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        this.beta = torch.zeros(new long[]{embedSize},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        // register_parameter ByRef: keep original handles after register
        register_parameter("alpha", alpha);
        register_parameter("beta", beta);
    }

    public int embedSize() {
        return embedSize;
    }

    public float eps() {
        return eps;
    }

    @Override
    public Tensor forward(Tensor x) {
        long dim = x.dim();
        long batchSize = x.size(0);

        if (dim == 2L) {
            // 2D: (batch, embed)
            Tensor bnOut = bn.forward(x);
            Tensor p = bnOut.mul(beta).sigmoid();
            Tensor alphaB = alpha.reshape(1L, embedSize).expand(batchSize, embedSize);
            // Dice: alpha * (1-p) * x + p * x  — Scala: term1 = alphaB.mul(p.neg().add(1)); term2 = p.mul(x); term1.add(term2)
            // Note: Scala term1 does not multiply by x; matches ported formula as written.
            Tensor term1 = alphaB.mul(p.neg().add(new Scalar(1.0)));
            Tensor term2 = p.mul(x);
            return term1.add(term2);
        } else {
            // 3D: (batch, seq, embed)
            long seqLen = x.size(1);
            Tensor xFlat = x.transpose(1, 2).reshape(batchSize * seqLen, embedSize);
            Tensor bnOut = bn.forward(xFlat);
            Tensor p = bnOut.mul(beta).sigmoid();
            Tensor alphaB = alpha.reshape(1L, embedSize).expand(batchSize * seqLen, embedSize);
            Tensor term1 = alphaB.mul(p.neg().add(new Scalar(1.0)));
            Tensor term2 = p.mul(xFlat);
            return term1.add(term2).reshape(batchSize, embedSize, seqLen).transpose(1, 2);
        }
    }
}
