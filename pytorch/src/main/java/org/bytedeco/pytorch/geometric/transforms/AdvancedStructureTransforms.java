/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * FeaturePropagation / HalfHop / AddGPSE
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/** Feature completion and structural upsampling. */
public final class AdvancedStructureTransforms {
    private AdvancedStructureTransforms() {}

    /**
     * Iterative feature propagation for missing (all-zero) node features.
     * Known rows are clamped back each iteration.
     */
    public static class FeaturePropagation implements BaseTransform {
        private final int numIterations;
        public FeaturePropagation(int numIterations) {
            if (numIterations <= 0) throw new IllegalArgumentException("numIterations > 0");
            this.numIterations = numIterations;
        }

        @Override
        public GraphData apply(GraphData data) {
            Tensor x0 = TransformUtils.requireX(data).clone();
            // known = row L2 norm > 0
            Tensor rowNorm = x0.pow(new Scalar(2)).sum(new long[]{1}).sqrt();
            Tensor known = rowNorm.gt(new Scalar(0.0)); // [N] bool

            data = new AdvancedStructuralTransforms.ToDense().apply(data);
            Tensor adj = data.adj;
            Tensor deg = adj.sum(1);
            Tensor dInv = deg.pow(new Scalar(-1.0));
            dInv.masked_fill_(deg.eq(new Scalar(0.0)), new Scalar(0.0));
            Tensor p = diag(dInv).matmul(adj);

            Tensor x = x0.clone();
            for (int i = 0; i < numIterations; i++) {
                x = p.matmul(x);
                // restore known features
                // index_put with bool mask on dim 0
                x.index_put_(new TensorIndexVector(new TensorIndex(known)),
                        x0.index(new TensorIndexVector(new TensorIndex(known))));
            }
            data.x = x;
            return data;
        }
    }

    /**
     * Half-Hop: insert a virtual node on every edge (u,v) → (u,d),(d,v).
     * Nodes: N → N+E ; edges: E → 2E.
     */
    public static class HalfHop implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            Tensor x = TransformUtils.requireX(data);
            Tensor ei = TransformUtils.requireEdgeIndex(data);
            long numNodes = x.size(0);
            Tensor row = ei.select(0, 0);
            Tensor col = ei.select(0, 1);
            long numEdges = row.size(0);

            Tensor vFeats = zeros(new long[]{numEdges, x.size(1)}, x.options());
            data.x = cat(new TensorVector(x, vFeats), 0);

            Tensor dummy = arange(new Scalar(numNodes), new Scalar(numNodes + numEdges),
                    TransformUtils.longOptsLike(ei));
            Tensor u2d = stack(new TensorVector(row, dummy), 0);
            Tensor d2v = stack(new TensorVector(dummy, col), 0);
            data.edge_index = cat(new TensorVector(u2d, d2v), 1);
            return data;
        }
    }

    /** Compose LapPE(k=8) + RWPE(walk=16). */
    public static class AddGPSE implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            data = new SpectralAndStructuralTransforms.AddLaplacianEigenvectorPE(8).apply(data);
            data = new SpectralAndStructuralTransforms.AddRandomWalkPE(16).apply(data);
            return data;
        }
    }
}
