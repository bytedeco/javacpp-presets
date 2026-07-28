package org.bytedeco.pytorch.geometric.attention;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * SGFormer simple global attention (Wu et al.).
 *
 * <pre>
 *   g = mean(X)
 *   α_i = σ( (q_i · k_g) / √d )
 *   y_i = α_i ⊙ v_g
 * </pre>
 * Linear complexity: one global token for K/V, all nodes as Q.
 */
public class SGFormerAttention extends Module {

    private final LinearImpl linQ;
    private final LinearImpl linK;
    private final LinearImpl linV;
    private final LinearImpl linOut;
    private final long numHeads;
    private final long headDim;
    private final long inChannels;

    public SGFormerAttention(long inChannels, long numHeads) {
        super();
        if (inChannels <= 0 || numHeads <= 0) {
            throw new IllegalArgumentException("inChannels/numHeads must be > 0");
        }
        if (inChannels % numHeads != 0) {
            throw new IllegalArgumentException("inChannels must be divisible by numHeads");
        }
        this.inChannels = inChannels;
        this.numHeads = numHeads;
        this.headDim = inChannels / numHeads;

        this.linQ = register_module("linQ", new LinearImpl(inChannels, inChannels));
        this.linK = register_module("linK", new LinearImpl(inChannels, inChannels));
        this.linV = register_module("linV", new LinearImpl(inChannels, inChannels));
        this.linOut = register_module("linOut", new LinearImpl(inChannels, inChannels));
    }

    /** @param x [N, C] @return [N, C] */
    public Tensor forward(Tensor x) {
        if (x == null || x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException("x must be [N," + inChannels + "]");
        }
        long N = x.size(0);
        long C = inChannels;

        // Global token
        Tensor g = x.mean(new long[]{0}, true, new ScalarTypeOptional()); // [1, C]

        Tensor q = linQ.forward(x).view(N, numHeads, headDim);   // [N,H,D]
        Tensor k = linK.forward(g).view(1, numHeads, headDim);   // [1,H,D]
        Tensor v = linV.forward(g).view(1, numHeads, headDim);   // [1,H,D]

        // Scaled dot-product with single global key → [N,H,1]
        Tensor score = q.mul(k).sum(new long[]{2}, true, new ScalarTypeOptional());
        score = score.mul(new Scalar(1.0 / Math.sqrt(headDim)));
        // Gating (single key makes softmax degenerate to 1)
        Tensor attn = torch.sigmoid(score);

        Tensor out = v.mul(attn).reshape(N, C); // [N,H,D] → [N,C]
        return linOut.forward(out);
    }

    public long getNumHeads() {
        return numHeads;
    }

    public long getInChannels() {
        return inChannels;
    }
}
