package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

/**
 * Parameter-free neighborhood aggregation (PyG {@code SimpleConv}).
 *
 * <pre>
 *   out_i = aggr_{j ∈ N(i)} (w_{ij} · x_j)
 *   optionally combine with root: sum | cat | self_loop
 * </pre>
 */
public class SimpleConv extends MessagePassing {

    private final String combineRoot; // "sum", "cat", "self_loop", or null

    public SimpleConv() {
        this("sum", null);
    }

    public SimpleConv(String aggr) {
        this(aggr, null);
    }

    /**
     * @param aggr        aggregation reduce: sum/mean/max/min/...
     * @param combineRoot how to fuse center node: {@code sum}, {@code cat},
     *                    {@code self_loop}, or {@code null} (neighbors only)
     */
    public SimpleConv(String aggr, String combineRoot) {
        super(aggr != null ? aggr : "sum");
        this.combineRoot = combineRoot;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor) null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        Tensor out = propagate(edge_index, x, edge_weight);
        if (combineRoot == null) {
            return out;
        }
        switch (combineRoot.toLowerCase()) {
            case "sum":
            case "self_loop":
                return out.add(x);
            case "cat":
                return torch.cat(new TensorVector(x, out), -1);
            default:
                throw new IllegalArgumentException(
                        "Unknown combineRoot='" + combineRoot + "' (use sum|cat|self_loop|null)");
        }
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        if (edge_attr != null) {
            if (edge_attr.dim() == 1) {
                return x_j.mul(edge_attr.view(new long[]{-1, 1}));
            }
            return x_j.mul(edge_attr);
        }
        return x_j;
    }

    public String getCombineRoot() {
        return combineRoot;
    }
}
