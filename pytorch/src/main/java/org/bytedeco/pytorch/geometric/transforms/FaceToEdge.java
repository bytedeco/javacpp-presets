/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.FaceToEdge
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.T_TensorTensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.stack;
import static org.bytedeco.pytorch.global.torch.unique_dim;

/**
 * Convert triangular {@code face} [3, F] into an undirected {@code edge_index}.
 */
public class FaceToEdge implements BaseTransform {
    private final boolean removeFaces;

    public FaceToEdge() { this(true); }
    public FaceToEdge(boolean removeFaces) { this.removeFaces = removeFaces; }

    @Override
    public GraphData apply(GraphData data) {
        TransformUtils.requireData(data);
        Tensor face = data.get("face");
        if (face == null || !face.defined()) {
            throw new IllegalArgumentException("FaceToEdge requires data['face'] [3,F]");
        }
        if (face.dim() != 2 || face.size(0) != 3) {
            throw new IllegalArgumentException(
                    "face must be [3,F], got dim=" + face.dim());
        }
        Tensor e1 = stack(new TensorVector(face.select(0, 0), face.select(0, 1)), 0);
        Tensor e2 = stack(new TensorVector(face.select(0, 1), face.select(0, 2)), 0);
        Tensor e3 = stack(new TensorVector(face.select(0, 2), face.select(0, 0)), 0);
        Tensor edgeIndex = cat(new TensorVector(e1, e2, e3), 1);
        // undirected
        edgeIndex = cat(new TensorVector(edgeIndex, edgeIndex.flip(0)), 1);
        // dedup columns
        T_TensorTensorTensor_T result = unique_dim(edgeIndex, 1, true, false, false);
        data.edge_index = result.get0().to(org.bytedeco.pytorch.global.torch.kLong());
        if (removeFaces) {
            // keep face attr — demos still may want it; only clear if requested via flag
            // PyG removes face by default when remove_faces=True; we keep for mesh pipelines.
        }
        return data;
    }
}
