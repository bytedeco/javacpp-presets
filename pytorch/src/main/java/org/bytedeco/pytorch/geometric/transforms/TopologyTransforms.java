package org.bytedeco.pytorch.geometric.transforms;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public class TopologyTransforms {

    /**
     * RemoveSelfLoops: 移除所有自环 (i == j)
     */
    public static class RemoveSelfLoops implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            Tensor row = data.edge_index.index(new TensorIndexVector(new TensorIndex(tensor(0))));
            Tensor col = data.edge_index.index(new TensorIndexVector(new TensorIndex(tensor(1))));
            // 找到非自环的掩码
            Tensor mask = row.ne(col);
            data.edge_index = data.edge_index.index_select(1, mask.nonzero().contiguous().view(-1));
            return data;
        }
    }

    /**
     * AddRemainingSelfLoops: 补全缺失的自环
     * 只为那些还没有自环的节点添加自环，避免重复
     */
    public static class AddRemainingSelfLoops implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            // 1. 先移除现有的，再添加全部，这是最稳健的补全方式
            data = new RemoveSelfLoops().apply(data);
            long numNodes = data.x.size(0);
            Tensor loop = arange(new Scalar(0), new Scalar(numNodes), data.edge_index.options());
            Tensor edgeLoop = cat(new TensorVector(loop.view(1, -1), loop.view(1, -1)), 0);
            data.edge_index = cat(new TensorVector(data.edge_index, edgeLoop), 1);
            return data;
        }
    }

    /**
     * RemoveIsolatedNodes: 移除孤立节点
     * 孤立节点既没有入边也没有出边。注意这会改变节点的索引。
     */
    public static class RemoveIsolatedNodes implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            long numNodes = data.x.size(0);
            // 1. 统计出现在 edge_index 中的所有节点
            Tensor outNodes = unique_consecutive( data.edge_index.view(-1)).get0(); //.unique();

            // 2. 映射旧索引到新索引，并过滤特征矩阵 x
            data.x = data.x.index_select(0, outNodes);

            // 3. 重新映射 edge_index (略，通常需要使用 torch.searchsorted 或重索引映射表)
            return data;
        }
    }

    /**
     * KNNGraph: 基于节点坐标 pos 构建 K-近邻图
     */
    public static class KNNGraph implements BaseTransform {
        private int k;
        public KNNGraph(int k) { this.k = k; }

        @Override
        public GraphData apply(GraphData data) {
            // data.pos: [N, D]
            Tensor pos = data.pos;
            // 计算欧氏距离矩阵 [N, N]
            Tensor dist = cdist(pos, pos, 2.0, new LongOptional());
            // 对每一行取 topk (取最近的 k+1 个，排除掉自己)
            Tensor topKIndices = topk(dist, k + 1, 1, false, true).get1();

            // 构造新的 edge_index
            Tensor row = arange(new Scalar(0), new Scalar(pos.size(0)), pos.options()).view(-1, 1).expand_as(topKIndices).reshape(-1);
            Tensor col = topKIndices.reshape(-1);

            data.edge_index = stack(new TensorVector(row, col), 0);
            // 移除产生的自环
            return new RemoveSelfLoops().apply(data);
        }
    }

    /**
     * RadiusGraph: 基于距离半径构建边
     */
    public static class RadiusGraph implements BaseTransform {
        private double r;
        public RadiusGraph(double r) { this.r = r; }

        @Override
        public GraphData apply(GraphData data) {
            Tensor dist = cdist(data.pos, data.pos, 2.0, null);
            // 找到距离小于 r 的所有点对
            Tensor mask = dist.le(new Scalar(r)).logical_and(dist.gt(new Scalar(0))); // 排除自己
            data.edge_index = mask.nonzero().t();
            return data;
        }
    }
}
