package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class GridSampling implements BaseTransform {
    private final float size;

    public GridSampling(float size) { this.size = size; }
    @Override
    public GraphData apply(GraphData data) {
        // 1. 计算体素坐标 ID [N, 3]
        // 使用 div(Scalar) 确保所有维度的体素大小一致
//        Tensor cluster = data.pos.div(size).floor_().to(kLong());
        Tensor cluster = data.pos.div(new Scalar(size)).floor_().to(kLong());
        // 2. 使用 unique_dim 对节点进行全局聚类
        // 参数说明：
        // cluster: 输入 Tensor
        // 0: 维度 dim=0 (按行/节点去重)
        // true: sorted=true (对体素 ID 排序)
        // true: return_inverse=true (我们需要这个来找回原始点到体素的映射)
        // false: return_counts=false
        T_TensorTensorTensor_T out = unique_dim(cluster, 0, true, true, false);

        // 唯一体素的坐标标识 (不直接用到，但可以存为 cluster_id)
        Tensor uniqueClusters = out.get0();
        // 每个原始点所属的唯一体素的索引 [N]
        Tensor inverseIdx = out.get1();

        // 3. 选取代表点
        // PyG 的默认实现通常是取每个体素内所有点的质心 (scatter_mean)
        // 这里的“取第一个点”实现，需要先拿到每个 cluster 第一次出现的下标

        // 为了高效拿到每个 unique 索引第一次出现的下标：
        // 我们利用 unique 对 inverseIdx 再做一次处理
        T_TensorTensorTensor_T representativeOut = unique_dim(inverseIdx, 0, true, false, false);
        // 这里得到的索引就是每个体素在原数据中的一个代表位置
        Tensor firstOccurrenceIdx = representativeOut.get0();

        // 4. 更新数据
        data.pos = data.pos.index_select(0, firstOccurrenceIdx);
        if (data.x != null) {
            data.x = data.x.index_select(0, firstOccurrenceIdx);
        }

        // 采样后，旧的 edge_index 已经失效，必须清除或重新构建 (例如通过 KNN)
        data.edge_index = null;

        return data;
    }

//    public GraphData apply(GraphData data) {
//        // 计算体素坐标 ID
//        Tensor cluster = data.pos.div(new Scalar(size)).floor_().to(kLong());
//        // 利用 unique 获取每个体素的唯一标识
//        T_TensorTensorTensor_T out = unique_consecutive(cluster, true, true, false);
//        Tensor inverseIdx = out.get1(); // 记录每个原点属于哪个体素
//
//        // 对每个 cluster 进行聚合（简单实现：取体素内第一个点）
//        // 进阶实现可使用 scatter_mean
//        data.pos = data.pos.index_select(0, unique_consecutive(inverseIdx, true, false, false).get0());
//        data.x = data.x.index_select(0, unique_consecutive(inverseIdx, true, false, false).get0());
//
//        return data;
//    }
}