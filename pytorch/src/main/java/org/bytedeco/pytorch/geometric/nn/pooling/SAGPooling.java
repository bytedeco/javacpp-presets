package org.bytedeco.pytorch.geometric.nn.pooling;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;
import org.bytedeco.pytorch.global.torch;

/**
 * SAGPooling — Self-Attention Graph Pooling (Lee et al. / PyG).
 *
 * <pre>
 *   score = tanh( GCN(x, edge_index) )     // structure-aware scores
 *   keep top ⌈ratio · N_g⌉ nodes per graph
 *   x'    = x ⊙ tanh(score)  for kept nodes
 * </pre>
 * Extends {@link TopKPooling}; only the score function differs (GCN vs projection).
 */
public class SAGPooling extends TopKPooling {

    private final GCNConv gnnScore;

    public SAGPooling(long inChannels, double ratio) {
        super(inChannels, ratio);
        // Score GCN: in → 1 (add self-loops + gcn_norm inside GCNConv)
        this.gnnScore = register_module("gnnScore", new GCNConv(inChannels, 1));
    }

    @Override
    protected Tensor calculateScore(Tensor x, Tensor edge_index) {
        // GCN outputs [N, 1]; squeeze to [N]
        Tensor s = gnnScore.forward(x, edge_index);
        if (s.dim() == 2 && s.size(1) == 1) {
            s = s.squeeze(1);
        }
        return torch.tanh(s);
    }

    /**
     * Explicit SAG pool entry (same as {@link #forwardGraph}).
     *
     * @return {@code {x_new, edge_index_new, batch_new, perm, score}}
     */
    public Tensor[] sagPool(Tensor x, Tensor edge_index, Tensor batch) {
        return forwardGraph(x, edge_index, batch);
    }

    public GCNConv getGnnScore() {
        return gnnScore;
    }
}
