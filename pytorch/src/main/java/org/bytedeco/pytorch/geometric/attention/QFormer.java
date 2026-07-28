package org.bytedeco.pytorch.geometric.attention;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.nn.norm.LayerNorm;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.MultiheadAttentionImpl;

/**
 * Q-Former (Querying Transformer): learnable queries cross-attend to input tokens.
 *
 * <pre>
 *   Q = learnable [nQ, D]
 *   H = LN(Q + CrossAttn(Q, X, X))
 *   Y = LN(H + FFN(H))
 * </pre>
 * Accepts {@code x} as {@code [N,D]} (treated as batch=1) or {@code [B,N,D]}.
 * Returns {@code [B, nQ, D]}.
 */
public class QFormer extends Module {

    private final Parameter queryTokens; // [nQ, D] owned leaf
    private final MultiheadAttentionImpl crossAttn;
    private final LinearImpl ffn1;
    private final LinearImpl ffn2;
    private final LayerNorm ln1;
    private final LayerNorm ln2;
    private final long dim;
    private final long numQueries;
    private final long numHeads;

    public QFormer(long dim, long numHeads, long numQueries) {
        super();
        if (dim <= 0 || numHeads <= 0 || numQueries <= 0) {
            throw new IllegalArgumentException("dim/numHeads/numQueries must be > 0");
        }
        if (dim % numHeads != 0) {
            throw new IllegalArgumentException("dim must be divisible by numHeads");
        }
        this.dim = dim;
        this.numHeads = numHeads;
        this.numQueries = numQueries;

        TensorOptions fOpt = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float));
        Tensor qInit = torch.randn(new long[]{numQueries, dim}, fOpt).clone();
        qInit.requires_grad_(true);
        this.queryTokens = new Parameter(qInit, true);
        register_parameter("queryTokens", this.queryTokens);

        this.crossAttn = register_module("crossAttn", new MultiheadAttentionImpl(dim, numHeads));
        this.ffn1 = register_module("ffn1", new LinearImpl(dim, dim * 4));
        this.ffn2 = register_module("ffn2", new LinearImpl(dim * 4, dim));
        this.ln1 = register_module("ln1", new LayerNorm(dim, 1e-12, true));
        this.ln2 = register_module("ln2", new LayerNorm(dim, 1e-12, true));
    }

    /**
     * @param x [N, D] or [B, N, D] input features (keys/values)
     * @return [B, numQueries, D]
     */
    public Tensor forward(Tensor x) {
        if (x == null) {
            throw new NullPointerException("x must not be null");
        }
        if (x.dim() == 2) {
            x = x.unsqueeze(0); // [1, N, D]
        }
        if (x.dim() != 3 || x.size(2) != dim) {
            throw new IllegalArgumentException(
                    "x must be [B,N," + dim + "] or [N," + dim + "], got dim=" + x.dim()
                            + " last=" + (x.dim() > 0 ? x.size(x.dim() - 1) : -1));
        }
        long B = x.size(0);

        // MHA layout: [L, B, D]
        Tensor mem = x.transpose(0, 1).contiguous();                          // [N, B, D]
        Tensor tgt = queryTokens.unsqueeze(1)
                .expand(new long[]{numQueries, B, dim}).contiguous();         // [nQ, B, D]

        T_TensorTensor_T mha = crossAttn.forwardT_TensorTensor_T(tgt, mem, mem);
        Tensor attnOut = mha.get0();                                          // [nQ, B, D]

        // Residual + LN  (work in [B, L, D])
        Tensor h = ln1.forward(tgt.add(attnOut).transpose(0, 1).contiguous());
        Tensor ffn = ffn2.forward(torch.relu(ffn1.forward(h)));
        return ln2.forward(h.add(ffn));                                       // [B, nQ, D]
    }

    public long getDim() {
        return dim;
    }

    public long getNumQueries() {
        return numQueries;
    }

    public long getNumHeads() {
        return numHeads;
    }

    public Parameter getQueryTokens() {
        return queryTokens;
    }
}
