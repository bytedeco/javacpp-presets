/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peers: RemoveSelfLoops / AddRemainingSelfLoops / KNNGraph / RadiusGraph
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/** Topology construction / cleanup transforms. */
public final class TopologyTransforms {
    private TopologyTransforms() {}

    /** Drop edges where {@code src == dst}. */
    public static class RemoveSelfLoops implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            Tensor ei = TransformUtils.requireEdgeIndex(data);
            Tensor[] pair = TransformUtils.removeSelfLoops(ei, data.edge_attr);
            data.edge_index = pair[0];
            if (pair[1] != null) data.edge_attr = pair[1];
            if (data.edge_weight != null && data.edge_weight.defined()) {
                Tensor[] wp = TransformUtils.removeSelfLoops(ei, data.edge_weight);
                data.edge_weight = wp[1];
            }
            return data;
        }
    }

    /** Ensure every node has exactly one self-loop (remove then re-add). */
    public static class AddRemainingSelfLoops implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            long n = TransformUtils.numNodes(data);
            if (data.edge_index == null || !data.edge_index.defined()) {
                data.edge_index = TransformUtils.addSelfLoops(
                        zeros(new long[]{2, 0}, TransformUtils.longOpts(
                                data.x != null ? data.x.device()
                                        : new org.bytedeco.pytorch.Device(
                                                org.bytedeco.pytorch.global.torch.DeviceType.CPU))), n);
                return data;
            }
            data.edge_index = TransformUtils.removeSelfLoops(data.edge_index);
            data.edge_index = TransformUtils.addSelfLoops(data.edge_index, n);
            return data;
        }
    }

    /** Build a k-NN graph from {@code pos} (excludes self). */
    public static class KNNGraph implements BaseTransform {
        private final int k;
        public KNNGraph(int k) {
            if (k <= 0) throw new IllegalArgumentException("k must be > 0");
            this.k = k;
        }

        @Override
        public GraphData apply(GraphData data) {
            Tensor pos = TransformUtils.requirePos(data);
            long n = pos.size(0);
            int actualK = (int) Math.min(k, Math.max(0, n - 1));
            if (actualK == 0) {
                data.edge_index = zeros(new long[]{2, 0}, TransformUtils.longOptsLike(pos));
                return data;
            }
            Tensor dist = cdist(pos, pos, 2.0, new LongOptional());
            // smallest distances → topk on negated dist, largest=true
            Tensor idx = topk(dist.neg(), actualK + 1, 1, true, true).get1();
            // drop self (col 0 is usually self); take next actualK
            idx = idx.slice(1, new LongOptional(1), new LongOptional(actualK + 1), 1);
            Tensor row = arange(new Scalar(0), new Scalar(n), TransformUtils.longOptsLike(pos))
                    .view(-1, 1).expand(new long[]{n, actualK}).reshape(-1);
            Tensor col = idx.reshape(-1).to(kLong());
            data.edge_index = stack(new TensorVector(row, col), 0);
            // safety: drop any residual self-loops
            data.edge_index = TransformUtils.removeSelfLoops(data.edge_index);
            return data;
        }
    }

    /** Connect all pairs with distance ≤ r (no self-loops). */
    public static class RadiusGraph implements BaseTransform {
        private final double r;
        public RadiusGraph(double r) {
            if (r <= 0) throw new IllegalArgumentException("r must be > 0");
            this.r = r;
        }

        @Override
        public GraphData apply(GraphData data) {
            Tensor pos = TransformUtils.requirePos(data);
            Tensor dist = cdist(pos, pos, 2.0, new LongOptional());
            Tensor mask = dist.le(new Scalar(r)).logical_and(dist.gt(new Scalar(0)));
            data.edge_index = mask.nonzero().t().to(kLong());
            return data;
        }
    }
}
