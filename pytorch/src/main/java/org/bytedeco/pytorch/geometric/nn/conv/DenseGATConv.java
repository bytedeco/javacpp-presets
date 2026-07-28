package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Dense (batched) multi-head GAT convolution.
 *
 * <pre>
 *   e_{ij}^{(h)} = LeakyReLU( a_src^{(h)}·Wh_i + a_dst^{(h)}·Wh_j )
 *   α_{ij}^{(h)} = softmax_j(e_{ij}^{(h)})   (masked by adj)
 *   y_i          = ∥_h Σ_j α_{ij}^{(h)} (Wh_j)^{(h)}
 * </pre>
 * Inputs {@code x [B,N,F_in]}, {@code adj [B,N,N]}. Output {@code [B,N,heads·F_out]}.
 */
public class DenseGATConv extends MessagePassing {

    private final LinearImpl lin;
    private final Parameter attSrc; // [1, H, C]
    private final Parameter attDst; // [1, H, C]
    private final long heads;
    private final long outChannels;
    private final long inChannels;
    private final double negativeSlope;
    private final boolean concat;

    public DenseGATConv(long inChannels, long outChannels, long heads) {
        this(inChannels, outChannels, heads, true, 0.2);
    }

    public DenseGATConv(long inChannels, long outChannels, long heads,
                        boolean concat, double negativeSlope) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0 || heads < 1) {
            throw new IllegalArgumentException("DenseGATConv dims invalid");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.heads = heads;
        this.concat = concat;
        this.negativeSlope = negativeSlope;

        this.lin = register_module("lin", new LinearImpl(inChannels, heads * outChannels));

        Tensor as = torch.randn(new long[]{1, heads, outChannels});
        torch.xavier_uniform_(as);
        this.attSrc = new Parameter(as.clone().requires_grad_(true), true);
        register_parameter("attSrc", this.attSrc);

        Tensor ad = torch.randn(new long[]{1, heads, outChannels});
        torch.xavier_uniform_(ad);
        this.attDst = new Parameter(ad.clone().requires_grad_(true), true);
        register_parameter("attDst", this.attDst);
    }

    /**
     * Dense forward. Second arg is adjacency {@code [B,N,N]}.
     * @param x   [B, N, inChannels]
     * @param adj [B, N, N] (0/1 or weighted; zeros are masked out of attention)
     * @return [B, N, heads*out] if concat else [B, N, out]
     */
    @Override
    public Tensor forward(Tensor x, Tensor adj) {
        if (x == null || adj == null) {
            throw new NullPointerException("x and adj must not be null");
        }
        if (x.dim() != 3 || adj.dim() != 3) {
            throw new IllegalArgumentException("x must be [B,N,C], adj [B,N,N]");
        }
        if (x.size(2) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(2)=" + x.size(2) + " != inChannels=" + inChannels);
        }
        long B = x.size(0);
        long N = x.size(1);

        // [B,N,H,C]
        Tensor xFeat = lin.forward(x).view(B, N, heads, outChannels);

        // a^T Wh → [B,N,H]
        Tensor alphaSrc = xFeat.mul(attSrc).sum(new long[]{3}, false, new ScalarTypeOptional());
        Tensor alphaDst = xFeat.mul(attDst).sum(new long[]{3}, false, new ScalarTypeOptional());

        // logits[b,i,j,h] = α_src[b,i,h] + α_dst[b,j,h]
        Tensor logits = alphaSrc.unsqueeze(2).add(alphaDst.unsqueeze(1));
        logits = torch.leaky_relu(logits, new Scalar(negativeSlope));

        // Mask non-edges
        Tensor mask = adj.unsqueeze(3).eq(new Scalar(0));
        logits = logits.masked_fill(mask, new Scalar(-1e9));

        // Softmax over neighbors (dim=2)
        Tensor alpha = torch.softmax(logits, 2); // [B,N,N,H]

        // Weighted messages: α[b,i,j,h] * xFeat[b,j,h,c] → sum_j
        // xFeat: [B,N,H,C] → [B,1,N,H,C]; alpha → [B,N,N,H,1]
        Tensor msg = xFeat.unsqueeze(1).mul(alpha.unsqueeze(4)); // [B,N,N,H,C]
        Tensor out = msg.sum(new long[]{2}, false, new ScalarTypeOptional()); // [B,N,H,C]

        if (concat) {
            return out.reshape(B, N, heads * outChannels);
        }
        return out.mean(2); // [B,N,C]
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }

    public long getHeads() {
        return heads;
    }

    public long getOutChannels() {
        return outChannels;
    }

    public boolean isConcat() {
        return concat;
    }

    public LinearImpl getLin() {
        return lin;
    }
}
