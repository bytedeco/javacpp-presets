package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndexVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * IndexToMask: 将索引张量转换为布尔掩码
 * [0, 2] -> [true, false, true, false...]
 */
public  class IndexToMask implements BaseTransform {
    private long size;
    public IndexToMask(long size) { this.size = size; }

    @Override
    public GraphData apply(GraphData data) {
        // 假设输入的 data.train_indices 是 [K] 形状的 LongTensor
        Tensor mask = zeros(new long[]{size}, data.x.options().dtype(new ScalarTypeOptional(kBool())));
        mask.index_put_(new TensorIndexVector(data.get("train_indices")), tensor(true, mask.options()));
        data.put("train_mask", mask);
        if (data.get("val_indices") != null) {
            mask = zeros(new long[]{size}, data.x.options().dtype(new ScalarTypeOptional(kBool())));
            mask.index_put_(new TensorIndexVector(data.get("val_indices")), tensor(true, mask.options()));
            data.put("val_mask", mask);
        }
        if (data.get("test_indices") != null) {
            mask = zeros(new long[]{size}, data.x.options().dtype(new ScalarTypeOptional(kBool())));
            mask.index_put_(new TensorIndexVector(data.get("test_indices")), tensor(true, mask.options()));
            data.put("test_mask", mask);
        }
        return data;
    }
}