package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;

public class ASAPooling extends Module {
    private GCNConv gnnScore; // 计算 Cluster Score
    private double ratio;

    public ASAPooling(long inChannels, double ratio) {
        this.ratio = ratio;
        this.gnnScore = new GCNConv(inChannels, 1);
        register_module("gnnScore", gnnScore);
    }

    public Tensor[] asaPool(Tensor x, Tensor edge_index, Tensor batch) {
        // 1. Cluster Formation (通常使用 LE 或 METIS，但在 End-to-End 中ASAP 提出了 Master 机制)
        // 这里为了简化，我们使用 ClusterPooling.graclus 生成初始簇
        long numNodes = x.size(0);
        Tensor cluster = ClusterPooling.graclus(edge_index, numNodes);

        // 2. Cluster Feature org.bytedeco.pytorch.geometric.aggr.Aggregation (Max + Mean)
        Tensor xPool = ClusterPooling.max_pool(cluster, x); // [NumClusters, C]

        // 3. Cluster Fitness Score Computation
        // ASAP 论文中，还需要考虑簇的拓扑。
        // 为了简化：我们构建粗化图，然后用 GCN 计算分数
        // 粗化图构建 (Coarsening Edge Index)
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);
        Tensor newRow = cluster.index_select(0, row);
        Tensor newCol = cluster.index_select(0, col);
        Tensor mask = newRow.ne(newCol);
        Tensor coarseedge_index = torch.stack(new TensorVector(newRow.masked_select(mask), newCol.masked_select(mask)), 0);

        // GNN Score on Coarse Graph
        Tensor fitness = gnnScore.forward(xPool, coarseedge_index).tanh().squeeze(1); // [NumClusters]

        // 4. TopK Selection on Clusters
        long numClusters = xPool.size(0);
        long k = (long) (numClusters * ratio);
        k = Math.max(1, k);

        T_TensorTensor_T topkRet = torch.topk(fitness, k);
        Tensor perm = topkRet.get1(); // Selected Cluster IDs
        Tensor score = topkRet.get0();

        // 5. Filter
        Tensor xFinal = xPool.index_select(0, perm).mul(score.unsqueeze(1)); // Gating

        // 更新 Batch
        Tensor batchPool = ClusterPooling.max_pool(cluster, batch.to(torch.ScalarType.Float)).to(torch.ScalarType.Long);
        Tensor batchFinal = batchPool.index_select(0, perm);

        // 更新 Edge Index (保留两端都在 perm 中的边)
        // (略：与 TopKPooling 中的 Relabel 逻辑一致)

        return new Tensor[]{xFinal, coarseedge_index, batchFinal};
    }
}