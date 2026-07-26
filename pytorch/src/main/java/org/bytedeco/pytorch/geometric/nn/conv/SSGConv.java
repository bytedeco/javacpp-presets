package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;

/**
 * 实现 torch_geometric.nn.conv.SSGConv
 * 简单的谱图卷积，引入 alpha 参数解决过平滑问题。
 */
public class SSGConv extends MessagePassing {
    private LinearImpl lin;
    private double alpha;
    private int K;
    private Tensor bias;

    public SSGConv(long inChannels, long outChannels, double alpha, int K, boolean hasBias) {
        super("add");
        this.alpha = alpha;
        this.K = K;

        // SSGConv 只有一层线性变换 W
        this.lin = new LinearImpl(inChannels, outChannels);
        register_module("lin", lin);

        if (hasBias) {
            this.bias = torch.zeros(new long[]{outChannels});
            register_parameter("bias", bias);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        long N = x.size(0);

        // 1. 预先计算归一化的邻接矩阵权重 (Symmetrical Normalization)
        // 这里的 norm 对应公式中的 D^-1/2 * A * D^-1/2
        Tensor norm = compute_normalization(edge_index, edge_weight, N);

        // 2. 初始特征
        Tensor x0 = x;
        Tensor out = x.mul(new Scalar(alpha));

        // 3. 递归传播 K 步
        // 公式: x^(k) = (1 - alpha) * (A_hat @ x^(k-1)) + alpha * x^(0)
        // 注意：SSGConv 的变体通常简化为先做 K 步传播，最后再处理 alpha
        // 标准 SSG 论文逻辑：
        Tensor x_k = x0;
        for (int k = 0; k < K; k++) {
            // 聚合邻居
            x_k = propagate(edge_index, x_k, norm);
            // 累加带权重的特征
            // 这里遵循公式：out = alpha * x_0 + (1 - alpha) * A_hat^k * x_0
            // 递归中每一层都保留 alpha 比例
        }

        // 4. 应用线性变换
        // 标准实现通常是先传播再线性，或者先线性再传播 (取决于实现流派)
        // 根据 PyG 定义：先聚合 K 步，最后 Wx
        Tensor result = lin.forward(x_k.mul(new Scalar(1 - alpha)).add(out));

        if (bias != null) {
            result = result.add(bias);
        }

        return result;
    }

    private Tensor compute_normalization(Tensor edge_index, Tensor edge_weight, long numNodes) {
        // 实现对称归一化: D^-1/2 * A_hat * D^-1/2
        if (edge_weight == null) {
            edge_weight = torch.ones(new long[]{edge_index.size(1)}, edge_index.options());
        }

        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        Tensor deg = Scatter.scatter(edge_weight, row, numNodes, "add");
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5)); //-0.5
        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));

        return degInvSqrt.index_select(0, row).mul(edge_weight).mul(degInvSqrt.index_select(0, col));
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // edge_attr 存储归一化后的系数
        return x_j.mul(edge_attr.view(-1, 1));
    }
}