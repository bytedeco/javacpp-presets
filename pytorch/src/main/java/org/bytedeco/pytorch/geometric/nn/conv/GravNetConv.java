package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 GravNetConv
 * 特点：动态构图 + 空间投影 + 高斯距离加权
 */
public class GravNetConv extends MessagePassing {
    private LinearImpl linSpatial;    // 用于投影到空间坐标 (S)
    private LinearImpl linFeature;    // 用于投影待传播特征 (F)
    private LinearImpl linOut;        // 最终聚合输出变换

    private int k;
    private int spaceDimensions;
    private int propagateDimensions;

    public GravNetConv(long inChannels, long outChannels, int spaceDimensions, int propagateDimensions, int k) {
        super("mean"); // 默认聚合方式
        this.k = k;
        this.spaceDimensions = spaceDimensions;
        this.propagateDimensions = propagateDimensions;

        // 1. 空间投影层 (S 维)
        this.linSpatial = new LinearImpl(inChannels, spaceDimensions);
        // 2. 特征投影层 (F 维)
        this.linFeature = new LinearImpl(inChannels, propagateDimensions);
        // 3. 输出拼接变换层
        this.linOut = new LinearImpl(propagateDimensions, outChannels);

        register_module("lin_spatial", linSpatial);
        register_module("lin_feature", linFeature);
        register_module("lin_out", linOut);
    }

    @Override
    public Tensor forward(Tensor x, Tensor batch) {
        long N = x.size(0);

        // --- 步骤 1: 投影 ---
        // s: 空间坐标 [N, S], 用于计算距离
        Tensor s = linSpatial.forward(x);
        // h: 传播特征 [N, F], 实际在图上传递的消息
        Tensor h = linFeature.forward(x);

        // --- 步骤 2: 动态构图 (k-NN) ---
        // 计算所有点对之间的欧几里得距离 [N, N]
        // 注意：在大规模图(N>10000)下此步极其耗显存，建议分块或使用专门的 knn 算子
        Tensor dists = torch.cdist(s, s);

        // 获取每个点的 k 个最近邻
        // topk 返回 (values, indices)，我们取最小的 k 个距离（注意排除自身通常在第一位）
        T_TensorTensor_T topk = dists.topk(k + 1, -1, false, true);
        Tensor knn_dist = topk.get0().narrow(-1, 1, k); // [N, k] 距离
        Tensor knn_idx = topk.get1().narrow(-1, 1, k);  // [N, k] 邻居索引

        // --- 步骤 3: 计算高斯权重 ---
        // w_ij = exp(-10 * d_ij^2)
        Tensor weights = knn_dist.pow(new Scalar(2)).mul(new Scalar(-10.0)).exp(); // [N, k]

        // --- 步骤 4: 消息传递 ---
        // 构建临时 edge_index 用于 MessagePassing 框架
        // targetIdx: [0,0,0... 1,1,1... N-1, N-1] 每个点重复 k 次
        Tensor targetIdx = torch.arange(new Scalar(N), knn_idx.options()).view(-1, 1).repeat(new long[]{1, k}).view(-1);
        Tensor sourceIdx = knn_idx.contiguous().view(-1);
        Tensor edge_index = torch.stack(new TensorVector(sourceIdx, targetIdx), 0);

        // 展平 weights 匹配 edge_index 维度 [N * k]
        Tensor edge_weights = weights.view(-1);

        // 调用 propagate
        Tensor out = propagate(edge_index, h, edge_weights);

        return linOut.forward(out);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_j: 邻居的 F 维特征
        // edge_attr: 高斯距离权重
        if (edge_attr != null) {
            return x_j.mul(edge_attr.view(-1, 1));
        }
        return x_j;
    }
}