/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.MaskToIndex
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

/** Convert train/val/test boolean masks to 1-D index tensors. */
public class MaskToIndex implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        TransformUtils.requireData(data);
        convert(data, "train_mask", "train_indices");
        convert(data, "val_mask", "val_indices");
        convert(data, "test_mask", "test_indices");
        return data;
    }

    private static void convert(GraphData data, String maskKey, String idxKey) {
        Tensor mask = data.get(maskKey);
        if (mask != null && mask.defined()) {
            data.put(idxKey, mask.nonzero().view(-1));
        }
    }
}
