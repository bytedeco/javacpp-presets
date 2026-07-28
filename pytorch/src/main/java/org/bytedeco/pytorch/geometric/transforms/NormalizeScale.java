/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.NormalizeScale
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.kFloat;

/** Center then scale {@code pos} into roughly {@code [-1, 1]}. */
public class NormalizeScale implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = TransformUtils.requirePos(data);
        pos = pos.sub(pos.mean(new long[]{0}, true, new ScalarTypeOptional(kFloat())));
        Tensor maxVal = pos.abs().max();
        data.pos = pos.div(maxVal.add(new Scalar(1e-7)));
        return data;
    }
}
