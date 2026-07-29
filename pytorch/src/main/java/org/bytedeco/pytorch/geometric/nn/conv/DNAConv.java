package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.NoGradGuard;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Dynamic Neighborhood Aggregation (DNAConv, Fey et al.).
 *
 * <p>Input is a multi-layer node tensor {@code x ∈ R^{N×L×C}}. For each historical
 * layer, keys/values are graph-smoothed via MessagePassing; attention over layers
 * then produces a single {@code [N, C]} output.
 *
 * <pre>
 *   Q,K,V = Linear(x)                         // per-layer projections
 *   K̂,V̂ = MP(K), MP(V)                       // topology smoothing per layer
 *   α = softmax_L( (Q_L · K̂) / √d )
 *   y = Σ_L α_L ⊙ V̂_L
 * </pre>
 */
public class DNAConv extends MessagePassing {

    private final long channels;
    private final int heads;
    private final int groups;
    private final long d_k;
    private final LinearImpl linQ;
    private final LinearImpl linK;
    private final LinearImpl linV;
    private final Parameter bias;

    public DNAConv(long channels, int heads, int groups, boolean hasBias) {
        super("sum");
        if (channels <= 0 || heads <= 0 || groups <= 0) {
            throw new IllegalArgumentException("channels/heads/groups must be > 0");
        }
        if (channels % heads != 0) {
            throw new IllegalArgumentException(
                    "channels must be divisible by heads: " + channels + "/" + heads);
        }
        this.channels = channels;
        this.heads = heads;
        this.groups = groups;
        this.d_k = channels / heads;

        this.linQ = register_module("lin_q", new LinearImpl(channels, heads * d_k));
        this.linK = register_module("lin_k", new LinearImpl(channels, heads * d_k));
        this.linV = register_module("lin_v", new LinearImpl(channels, heads * d_k));

        if (hasBias) {
            Tensor b = torch.zeros(new long[]{channels},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
            this.bias = new Parameter(b, true);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    /**
     * DNA expects multi-layer features. For API compatibility with
     * {@link MessagePassing#forward(Tensor, Tensor)}, a rank-2 input is treated
     * as a single layer {@code [N,1,C]}.
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        if (x.dim() == 2) {
            // [N,C] → [N,1,C]
            return forward(x.unsqueeze(1), edge_index, (Tensor) null);
        }
        return forward(x, edge_index, (Tensor) null);
    }

    /**
     * @param x           [N, L, C] multi-layer node features (or [N,C] via 2-arg)
     * @param edge_index  [2, E]
     * @param edge_weight optional [E] edge weights for topology smoothing
     * @return [N, C]
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (x.dim() == 2) {
            x = x.unsqueeze(1);
        }
        if (x.dim() != 3) {
            throw new IllegalArgumentException("x must be [N,L,C], dim=" + x.dim());
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2,E]");
        }
        long N = x.size(0);
        long L = x.size(1);
        long C = x.size(2);
        if (C != channels) {
            throw new IllegalArgumentException(
                    "x channels " + C + " != DNAConv.channels " + channels);
        }

        // Q/K/V projections on flattened layers
        Tensor xFlat = x.reshape(N * L, C);
        Tensor Q = linQ.forward(xFlat).view(N, L, heads, d_k);
        Tensor K = linK.forward(xFlat).view(N, L, heads, d_k);
        Tensor V = linV.forward(xFlat).view(N, L, heads, d_k);

        // Topology-smooth each layer's K and V via MessagePassing
        Tensor Khat = torch.zeros_like(K);
        Tensor Vhat = torch.zeros_like(V);
        for (long l = 0; l < L; l++) {
            Tensor Kl = K.select(1, l); // [N, H, d]
            Tensor Vl = V.select(1, l);
            // Homogeneous + edge_weight: use (edge, x, edgeAttr) overload
            Tensor Kh = super.propagate(edge_index, Kl, edge_weight);
            Tensor Vh = super.propagate(edge_index, Vl, edge_weight);
            Khat.select(1, l).copy_(Kh);
            Vhat.select(1, l).copy_(Vh);
        }

        // Cross-layer attention with query = last layer
        Tensor query = Q.select(1, L - 1).unsqueeze(1); // [N,1,H,d]
        Tensor attn = query.mul(Khat).sum(-1);          // [N,L,H]
        attn = attn.div(new Scalar(Math.sqrt(d_k)));
        attn = torch.softmax(attn, 1);

        Tensor out = attn.unsqueeze(-1).mul(Vhat).sum(1); // [N,H,d]
        Tensor res = out.reshape(N, channels);
        if (bias != null) {
            res = res.add(bias);
        }
        return res;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_j: [E, H, d] (or higher); edge_attr: [E] weights
        if (edge_attr != null) {
            Tensor w = edge_attr;
            while (w.dim() < x_j.dim()) {
                w = w.unsqueeze(-1);
            }
            return x_j.mul(w);
        }
        return x_j;
    }

    public void reset_parameters() {
        linQ.reset_parameters();
        linK.reset_parameters();
        linV.reset_parameters();
        if (bias != null) {
            try (NoGradGuard guard = new NoGradGuard()) {
                bias.fill_(new Scalar(0));
            }
        }
    }

    public long getChannels() {
        return channels;
    }

    public int getHeads() {
        return heads;
    }

    public int getGroups() {
        return groups;
    }

    /** Per-head key/query dimension (= channels / heads). */
    public long getDk() {
        return d_k;
    }

    public LinearImpl getLinQ() {
        return linQ;
    }

    public LinearImpl getLinK() {
        return linK;
    }

    public LinearImpl getLinV() {
        return linV;
    }
}
