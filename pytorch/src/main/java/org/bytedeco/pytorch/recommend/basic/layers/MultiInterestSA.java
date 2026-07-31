/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/MultiInterestSA.scala
 *
 * Self-attention multi-interest module (Comirec).
 * Input: seqEmb (B, L, D), mask (B, L, 1)
 * Output: (B, interest_num, D)
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MultiInterestSA extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embeddingDim;
    private final int interestNum;
    private final int actualHiddenDim;
    private final Tensor w1;
    private final Tensor w2;
    private final Tensor w3;

    public MultiInterestSA(int embeddingDim, int interestNum) {
        this(embeddingDim, interestNum, null, DeviceSupport.backend());
    }

    public MultiInterestSA(int embeddingDim, int interestNum, Integer hiddenDim, String device) {
        super("MultiInterestSA");
        this.embeddingDim = embeddingDim;
        this.interestNum = interestNum;
        this.actualHiddenDim = hiddenDim != null ? hiddenDim : embeddingDim * 4;

        TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        this.w1 = torch.rand(new long[]{embeddingDim, actualHiddenDim}, opts);
        this.w2 = torch.rand(new long[]{actualHiddenDim, interestNum}, opts);
        this.w3 = torch.rand(new long[]{embeddingDim, embeddingDim}, opts);

        register_parameter("w1", w1);
        register_parameter("w2", w2);
        register_parameter("w3", w3);
    }

    public Tensor forward(Tensor seqEmb, Tensor mask) {
        // H = seq_emb @ W1
        Tensor h = torch.matmul(seqEmb, w1).tanh();

        // A = H @ W2
        Tensor a = torch.matmul(h, w2);

        if (mask != null) {
            // Apply mask with large negative value (mirrors Scala)
            Tensor maskedA = a.add(mask.mul(new Scalar(-1e9f)).add(new Scalar(1e9f)));
            a = torch.softmax(maskedA, 1);
        } else {
            a = torch.softmax(a, 1);
        }

        // A: (batch, interest, seq) after transpose
        Tensor aTransposed = a.transpose(1, 2);

        // multi_interest_emb: (batch, interest, D)
        return torch.matmul(aTransposed, seqEmb);
    }

    @Override
    public Tensor forward(Tensor seqEmb) {
        return forward(seqEmb, (Tensor) null);
    }
}
