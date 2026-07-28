/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.AddSelfLoops
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

/**
 * Append self-loops {@code (i,i)} for every node.
 * Does not dedup existing loops — pair with {@link TopologyTransforms.RemoveSelfLoops}
 * or use {@link TopologyTransforms.AddRemainingSelfLoops}.
 */
public class AddSelfLoops implements BaseTransform {

    private final double fillValue;

    public AddSelfLoops() {
        this(1.0);
    }

    /** fillValue applied to new loop entries of edge_weight / edge_attr when present. */
    public AddSelfLoops(double fillValue) {
        this.fillValue = fillValue;
    }

    @Override
    public GraphData apply(GraphData data) {
        long n = TransformUtils.numNodes(data);
        Tensor ei = data.edge_index;
        if (ei == null || !ei.defined()) {
            // no edges yet — create pure self-loop graph
            data.edge_index = TransformUtils.addSelfLoops(
                    org.bytedeco.pytorch.global.torch.zeros(
                            new long[]{2, 0},
                            TransformUtils.longOpts(
                                    data.x != null ? data.x.device()
                                            : new org.bytedeco.pytorch.Device(
                                                    org.bytedeco.pytorch.global.torch.DeviceType.CPU))),
                    n);
            return data;
        }
        data.edge_index = TransformUtils.addSelfLoops(ei, n);
        // edge_weight: append fillValue for each new loop
        if (data.edge_weight != null && data.edge_weight.defined()) {
            Tensor loopW = org.bytedeco.pytorch.global.torch.full(
                    new long[]{n},
                    new org.bytedeco.pytorch.Scalar(fillValue),
                    data.edge_weight.options());
            data.edge_weight = org.bytedeco.pytorch.global.torch.cat(
                    new org.bytedeco.pytorch.TensorVector(data.edge_weight, loopW), 0);
        }
        return data;
    }
}
