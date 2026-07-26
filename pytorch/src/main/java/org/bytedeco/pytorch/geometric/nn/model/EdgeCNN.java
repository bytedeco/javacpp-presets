package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.EdgeConv;
//import static org.bytedeco.pytorch.geometric.nn.pooling.ClusterPooling.*;


public class EdgeCNN extends Module {
    private EdgeConv conv1, conv2;
    private LinearImpl classifier;
    private int k;

    public EdgeCNN(long inChannels, long outChannels, int k) {
        this.k = k;
        // org.bytedeco.pytorch.geometric.nn.conv.EdgeConv 内部 MLP: 2*In -> 64
        this.conv1 = new EdgeConv(inChannels, 64);
        this.conv2 = new EdgeConv(64, outChannels);

        register_module("conv1", conv1);
        register_module("conv2", conv2);
    }

    public Tensor forward(Tensor x, Tensor batch) {
        // Layer 1: Dynamic Graph -> Conv
        Tensor edge_index1 = knn_graph(x, k, batch);
        Tensor x1 = conv1.forward(x, edge_index1).relu();

        // Layer 2: Dynamic Graph on new features -> Conv
        Tensor edge_index2 = knn_graph(x1, k, batch);
        Tensor x2 = conv2.forward(x1, edge_index2).relu();

        // Global Max Pooling (Readout)
        // 假设 PointOps.max_pool 或 GlobalPooling 可用
        // return global_max_pool(x2, batch);
        return x2; // 返回节点特征用于分割，或池化后用于分类
    }

    /**
     * Exact K-Nearest Neighbors
     *
     * @param x 查询点 [N, D]
     * @param y 数据库点 [M, D] (若为null则 y=x)
     * @param k 邻居数
     * @return Tensor indices [N, k]
     */
    public static Tensor knn(Tensor x, Tensor y, int k, Tensor batchX, Tensor batchY) {
        if (y == null) y = x;

        // 1. 计算距离矩阵 [N, M]
        // cdist 计算 L2 距离
        Tensor dist = torch.cdist(x, y);

        // 2. 处理 Batch (Masking)
        if (batchX != null && batchY != null) {
            // mask[i, j] = True if batchX[i] != batchY[j]
            Tensor mask = batchX.unsqueeze(1).ne(batchY.unsqueeze(0));
            // 将不同 batch 的距离设为无穷大
            dist.masked_fill_(mask, new Scalar(Float.POSITIVE_INFINITY));
        }

        // 3. TopK (smallest)
        // largest=false -> smallest
        Tensor indices = torch.topk(dist, k, 1, false, true).get1();
        return indices;
    }

    public static Tensor knn_graph(Tensor x, int k, Tensor batch) {
        long N = x.size(0);

        // 1. Get Indices [N, k]
        Tensor col = knn(x, null, k, batch, batch);

        // 2. Create Row indices [N, k]
        // 0,0,0... 1,1,1...
        Tensor row = torch.arange(new Scalar(N), x.options()).unsqueeze(1).expand(new long[]{N, k});

        // 3. Flatten & Stack
        row = row.reshape(N * k);
        col = col.reshape(N * k);

        return torch.stack(new TensorVector(row, col), 0);
    }
}