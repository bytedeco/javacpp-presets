package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.geometric.data.GraphData;

/**
 * MaskToIndex: 将布尔掩码转换为索引张量
 * [true, false, true] -> [0, 2]
 */
public class MaskToIndex implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        if (data.get("train_mask") != null) {
            // nonzero() 返回的是 [K, 1]，通过 view(-1) 转为一维索引
            data.put("train_indices", data.get("train_mask").nonzero().view(-1));
        }
        if (data.get("val_mask") != null) {
            data.put("val_indices", data.get("val_mask").nonzero().view(-1));
        }
        if (data.get("test_mask") != null) {
            data.put("test_indices", data.get("test_mask").nonzero().view(-1));
        }
        return data;
    }
}