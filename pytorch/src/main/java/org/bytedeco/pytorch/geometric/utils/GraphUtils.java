package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

/**
 * Graph topology / normalization helpers shared by MessagePassing layers.
 */
public final class GraphUtils {

    private GraphUtils() {}

    /**
     * Add self-loops: {@code A_hat = A + I}.
     *
     * @param edge_index [2, E]
     * @param numNodes   N
     * @return [2, E + N]
     */
    public static Tensor add_self_loops(Tensor edge_index, long numNodes) {
        Tensor loop_index = torch.arange(new Scalar(0), new Scalar(numNodes), edge_index.options());
        loop_index = loop_index.unsqueeze(0).repeat(new long[]{2, 1});
        return torch.cat(new TensorVector(edge_index, loop_index), 1);
    }

    /**
     * Add self-loops and extend edge weights with {@code fillValue} on the new loops.
     *
     * @return {edge_index_with_loops, edge_weight_with_loops}
     */
    public static Tensor[] add_self_loops(Tensor edge_index, Tensor edge_weight,
                                         long numNodes, double fillValue) {
        Tensor ei = add_self_loops(edge_index, numNodes);
        Tensor loopW = torch.full(
                new long[]{numNodes},
                new Scalar(fillValue),
                edge_weight != null ? edge_weight.options()
                        : new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
        Tensor ew = edge_weight == null
                ? torch.cat(new TensorVector(
                        torch.ones(new long[]{edge_index.size(1)}, loopW.options()),
                        loopW), 0)
                : torch.cat(new TensorVector(edge_weight, loopW), 0);
        return new Tensor[]{ei, ew};
    }

    /**
     * Remove self-loops from edge_index.
     *
     * @return filtered edge_index [2, E']
     */
    public static Tensor remove_self_loops(Tensor edge_index) {
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);
        Tensor mask = row.ne(col);
        return edge_index.masked_select(mask.unsqueeze(0).expand_as(edge_index))
                .view(2, -1);
    }

    /**
     * Remove self-loops and filter a 1-D edge attribute in lockstep.
     *
     * @return {edge_index, edge_attr}
     */
    public static Tensor[] remove_self_loops(Tensor edge_index, Tensor edge_attr) {
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);
        Tensor mask = row.ne(col);
        Tensor ei = edge_index.masked_select(mask.unsqueeze(0).expand_as(edge_index)).view(2, -1);
        Tensor ea = edge_attr == null ? null : edge_attr.masked_select(mask);
        return new Tensor[]{ei, ea};
    }

    /**
     * Degree of nodes referenced by {@code index} [E] → [numNodes].
     */
    public static Tensor degree(Tensor index, long numNodes) {
        return degree(index, numNodes, torch.ScalarType.Float);
    }

    public static Tensor degree(Tensor index, long numNodes, torch.ScalarType dtype) {
        index = AggrUtils.asLongIndex(index);
        Tensor ones = torch.ones(
                new long[]{index.size(0)},
                new TensorOptions().dtype(new ScalarTypeOptional(dtype)));
        Tensor out = torch.zeros(
                new long[]{numNodes},
                new TensorOptions().dtype(new ScalarTypeOptional(dtype)));
        return out.index_add_(0, index, ones);
    }

    /**
     * Symmetric GCN normalization edge weights:
     * {@code norm_ij = deg(i)^{-1/2} * w_ij * deg(j)^{-1/2}}.
     *
     * <p>When {@code edge_weight == null}, treats all edges as weight 1.
     * Does <b>not</b> add self-loops itself — pass {@code addSelfLoops=true} to do so.
     *
     * @return {edge_index, edge_weight} (edge_index may gain self-loops)
     */
    public static Tensor[] gcn_norm(Tensor edge_index, Tensor edge_weight, long numNodes,
                                   boolean addSelfLoops, torch.ScalarType dtype) {
        if (addSelfLoops) {
            if (edge_weight == null) {
                edge_index = add_self_loops(edge_index, numNodes);
            } else {
                Tensor[] pair = add_self_loops(edge_index, edge_weight, numNodes, 1.0);
                edge_index = pair[0];
                edge_weight = pair[1];
            }
        }

        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        if (edge_weight == null) {
            edge_weight = torch.ones(
                    new long[]{edge_index.size(1)},
                    new TensorOptions().dtype(new ScalarTypeOptional(dtype)));
        } else if (edge_weight.scalar_type().intern() != dtype.intern()) {
            edge_weight = edge_weight.to(dtype);
        }

        // Degree from target side (col) for undirected-style GCN; use weighted degree.
        Tensor deg = torch.zeros(new long[]{numNodes}, edge_weight.options());
        deg.index_add_(0, col, edge_weight);

        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));

        Tensor norm = degInvSqrt.index_select(0, row)
                .mul(edge_weight)
                .mul(degInvSqrt.index_select(0, col));
        return new Tensor[]{edge_index, norm};
    }

    /** Convenience: gcn_norm with self-loops and Float dtype. */
    public static Tensor[] gcn_norm(Tensor edge_index, long numNodes) {
        return gcn_norm(edge_index, null, numNodes, true, torch.ScalarType.Float);
    }

    /** Segment softmax over edge groups (attention). */
    public static Tensor softmax(Tensor src, Tensor index, long numNodes) {
        return AggrUtils.scatter_softmax(src, index, numNodes);
    }

    /**
     * Infer bipartite size [N_src, N_dst] from features and/or edge endpoints.
     */
    public static long[] bipartite_size(Tensor edge_index, Tensor xSrc, Tensor xDst) {
        long nSrc = xSrc != null ? xSrc.size(0) : -1;
        long nDst = xDst != null ? xDst.size(0) : -1;
        if (nSrc < 0 || nDst < 0) {
            Tensor row = edge_index.select(0, 0);
            Tensor col = edge_index.select(0, 1);
            if (nSrc < 0) {
                nSrc = row.size(0) == 0 ? 0 : row.max().item_long() + 1;
            }
            if (nDst < 0) {
                nDst = col.size(0) == 0 ? 0 : col.max().item_long() + 1;
            }
        }
        return new long[]{nSrc, nDst};
    }
}
