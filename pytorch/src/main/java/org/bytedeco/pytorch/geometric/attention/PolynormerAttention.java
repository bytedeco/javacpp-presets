package org.bytedeco.pytorch.geometric.attention;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AttentionUtils;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Polynormer / linear attention with {@code φ(x) = elu(x) + 1}.
 *
 * <pre>
 *   y = φ(Q) (φ(K)ᵀ V) / (φ(Q) (φ(K)ᵀ 1))
 * </pre>
 * Linear complexity in sequence length; no random features required.
 */
public class PolynormerAttention extends Module {

    private final LinearImpl linQ;
    private final LinearImpl linK;
    private final LinearImpl linV;
    private final LinearImpl linOut;
    private final long numHeads;
    private final long headDim;
    private final long inChannels;

    public PolynormerAttention(long inChannels, long numHeads) {
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

        Tensor q = linQ.forward(x).view(N, numHeads, headDim);
        Tensor k = linK.forward(x).view(N, numHeads, headDim);
        Tensor v = linV.forward(x).view(N, numHeads, headDim);

        Tensor qP = AttentionUtils.kernel_elu(q).permute(1, 0, 2); // [H,N,D]
        Tensor kP = AttentionUtils.kernel_elu(k).permute(1, 0, 2);
        Tensor vP = v.permute(1, 0, 2);

        Tensor kv = kP.permute(0, 2, 1).matmul(vP); // [H,D,D]
        Tensor out = qP.matmul(kv);
        Tensor kSum = kP.sum(new long[]{1}, false, new ScalarTypeOptional()).unsqueeze(2);
        Tensor norm = qP.matmul(kSum).add(new Scalar(1e-6));
        out = out.div(norm).permute(1, 0, 2).reshape(N, C);
        return linOut.forward(out);
    }

    public long getNumHeads() {
        return numHeads;
    }

    public long getInChannels() {
        return inChannels;
    }
}
