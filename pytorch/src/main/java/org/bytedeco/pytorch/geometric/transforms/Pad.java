package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Pad: 填充特征或邻接矩阵以对齐 Batch 形状
 */
public class Pad implements BaseTransform {
    private long targetSize;
    public Pad(long targetSize) { this.targetSize = targetSize; }

    @Override
    public GraphData apply(GraphData data) {
        long currentSize = data.x.size(0);
        if (currentSize < targetSize) {
            long padLen = targetSize - currentSize;
            // 对特征矩阵 X 进行填充 [N, D] -> [targetSize, D]
            Tensor padding = zeros(new long[]{padLen, data.x.size(1)}, data.x.options());
            data.x = cat(new TensorVector(data.x, padding), 0);
        }
        return data;
    }
}