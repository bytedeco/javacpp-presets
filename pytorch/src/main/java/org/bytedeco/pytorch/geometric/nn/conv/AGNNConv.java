package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;

/**
 * Attention-based Graph Neural Network convolution (Thekumparampil et al.).
 *
 * <pre>
 *   cos_{ij} = (h_i · h_j) / (‖h_i‖ ‖h_j‖)
 *   α_{ij}   = softmax_j( β · cos_{ij} )
 *   h'_i     = Σ_j α_{ij} h_j
 * </pre>
 * β is a scalar temperature (learnable or fixed).
 */
public class AGNNConv extends MessagePassing {

    private final Parameter betaParam; // learnable β (null if fixed)
    private final Tensor betaFixed;    // non-learnable buffer handle
    private final boolean learnBeta;

    public AGNNConv() {
        this(true);
    }

    public AGNNConv(boolean requiresGrad) {
        super("sum");
        this.learnBeta = requiresGrad;
        Tensor betaInit = torch.ones(new long[]{1});
        if (requiresGrad) {
            // Own storage; register for bookkeeping only (ByRef-safe)
            Tensor leaf = betaInit.clone();
            leaf.requires_grad_(true);
            this.betaParam = new Parameter(leaf, true);
            register_parameter("beta", this.betaParam);
            this.betaFixed = null;
        } else {
            this.betaParam = null;
            this.betaFixed = betaInit.clone();
            register_buffer("beta", this.betaFixed);
        }
    }

    private Tensor beta() {
        return learnBeta ? betaParam : betaFixed;
    }

    @Override
    protected boolean needsX_i() {
        return true;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        return propagate(edge_index, x);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // cos = (x_i · x_j) / (‖x_i‖ ‖x_j‖)
        Tensor dot = x_i.mul(x_j).sum(-1); // [E] or [E, ...] last-dim reduced
        Tensor ni = x_i.norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, false)
                .clamp_min(new Scalar(1e-12));
        Tensor nj = x_j.norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, false)
                .clamp_min(new Scalar(1e-12));
        Tensor cos = dot.div(ni.mul(nj));
        Tensor logits = cos.mul(beta());

        Tensor targetIdx = _index_i != null ? _index_i : edge_index.select(0, 1);
        targetIdx = AggrUtils.asLongIndex(targetIdx);
        long n = numNodes > 0 ? numNodes : (_size != null ? _size[1] : 0);
        Tensor alpha = AggrUtils.scatter_softmax(logits, targetIdx, n);

        // Broadcast alpha onto feature dims
        while (alpha.dim() < x_j.dim()) {
            alpha = alpha.unsqueeze(-1);
        }
        return x_j.mul(alpha);
    }

    public void resetParameters() {
        Tensor b = beta();
        if (b != null) {
            b.fill_(new Scalar(1.0));
        }
    }

    public Tensor getBeta() {
        return beta();
    }

    public boolean isLearnBeta() {
        return learnBeta;
    }
}
