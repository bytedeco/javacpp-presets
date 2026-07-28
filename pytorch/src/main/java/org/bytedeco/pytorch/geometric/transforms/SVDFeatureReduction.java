/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.SVDFeatureReduction
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.StringViewOptional;
import org.bytedeco.pytorch.T_TensorTensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.linalg_svd;

/**
 * Project node features onto the top-{@code outChannels} left singular vectors
 * scaled by singular values (PyG {@code SVDFeatureReduction}).
 */
public class SVDFeatureReduction implements BaseTransform {

    private final int outChannels;

    public SVDFeatureReduction(int outChannels) {
        if (outChannels <= 0) {
            throw new IllegalArgumentException("outChannels must be > 0");
        }
        this.outChannels = outChannels;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor x = TransformUtils.requireX(data);
        long feat = x.size(1);
        int k = (int) Math.min(outChannels, Math.min(x.size(0), feat));
        // full_matrices=false → U is [N, min(N,F)], S is [min(N,F)]
        T_TensorTensorTensor_T svd = linalg_svd(x, /*full_matrices=*/false, new StringViewOptional());
        Tensor u = svd.get0();
        Tensor s = svd.get1();
        data.x = u.narrow(1, 0, k).mul(s.narrow(0, 0, k));
        return data;
    }
}
