/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peers: ToDense / TwoHop / GCNNorm / SIGN
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/** Structural / spectral-prep transforms used by SIGN pipelines. */
public final class AdvancedStructuralTransforms {
    private AdvancedStructuralTransforms() {}

    /** Sparse edge_index → dense adjacency {@code data.adj} [N,N]. */
    public static class ToDense implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            long n = TransformUtils.numNodes(data);
            Tensor ei = TransformUtils.requireEdgeIndex(data);
            Tensor ref = data.x != null ? data.x
                    : (data.edge_weight != null ? data.edge_weight
                    : ones(new long[]{1}, TransformUtils.floatOptsLike(ei)));
            Tensor adj = zeros(new long[]{n, n}, ref.options());
            Tensor vals = data.edge_weight != null && data.edge_weight.defined()
                    ? data.edge_weight.to(adj.dtype())
                    : ones(new long[]{ei.size(1)}, adj.options());
            adj.index_put_(
                    new TensorIndexVector(
                            new TensorIndex(ei.select(0, 0)),
                            new TensorIndex(ei.select(0, 1))),
                    vals);
            data.adj = adj;
            return data;
        }
    }

    /** Add 2-hop edges via dense A² &gt; 0 (small graphs). */
    public static class TwoHop implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            data = new ToDense().apply(data);
            Tensor twoHop = data.adj.matmul(data.adj).gt(new Scalar(0)).to(kLong());
            // keep original 1-hop as well
            Tensor combined = data.adj.gt(new Scalar(0)).to(kLong()).add(twoHop).gt(new Scalar(0));
            combined.fill_diagonal_(new Scalar(0));
            data.edge_index = combined.nonzero().t().to(kLong());
            return data;
        }
    }

    /**
     * GCN normalization: add self-loops then
     * {@code edge_weight = deg(i)^{-1/2} * deg(j)^{-1/2}}.
     */
    public static class GCNNorm implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            long n = TransformUtils.numNodes(data);
            Tensor ei = TransformUtils.requireEdgeIndex(data);
            Tensor loopEi = TransformUtils.addSelfLoops(
                    TransformUtils.removeSelfLoops(ei), n);

            Tensor ref = data.x != null ? data.x
                    : ones(new long[]{1}, TransformUtils.floatOptsLike(loopEi));
            Tensor deg = TransformUtils.degree(loopEi.select(0, 1), n, ref);
            Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
            degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));

            Tensor row = loopEi.select(0, 0);
            Tensor col = loopEi.select(0, 1);
            Tensor ew = degInvSqrt.index_select(0, row).mul(degInvSqrt.index_select(0, col));
            data.edge_index = loopEi;
            data.edge_weight = ew;
            return data;
        }
    }

    /**
     * SIGN: concatenate {@code [X, AX, A²X, … AᵏX]} under GCN-normalized A.
     */
    public static class SIGN implements BaseTransform {
        private final int k;
        public SIGN(int k) {
            if (k < 0) throw new IllegalArgumentException("k must be >= 0");
            this.k = k;
        }

        @Override
        public GraphData apply(GraphData data) {
            TransformUtils.requireX(data);
            data = new GCNNorm().apply(data);
            java.util.List<Tensor> xs = new java.util.ArrayList<>();
            Tensor cur = data.x;
            xs.add(cur);
            for (int i = 0; i < k; i++) {
                cur = aggregate(cur, data.edge_index, data.edge_weight);
                xs.add(cur);
            }
            data.x = cat(new TensorVector(xs.toArray(new Tensor[0])), 1);
            return data;
        }

        private static Tensor aggregate(Tensor x, Tensor edgeIndex, Tensor edgeWeight) {
            Tensor row = edgeIndex.select(0, 0);
            Tensor col = edgeIndex.select(0, 1);
            Tensor w = edgeWeight.view(new long[]{-1, 1});
            Tensor msg = x.index_select(0, row).mul(w);
            Tensor out = zeros_like(x);
            Tensor scatterIdx = col.unsqueeze(-1).expand_as(msg);
            out.scatter_add_(0, scatterIdx, msg);
            return out;
        }
    }
}
