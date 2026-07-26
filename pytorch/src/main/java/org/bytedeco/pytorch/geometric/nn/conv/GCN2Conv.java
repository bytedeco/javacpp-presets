package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.GCN2Conv
 * 解决了深层图卷积网络中的过平滑问题。
 */
public class GCN2Conv extends MessagePassing {
    private LinearImpl lin;         // 权重矩阵 W
    private LinearImpl linRes;      // 针对 GCNII* 非共享权重的 W_2
    private float alpha;            // 初始残差权重
    private float beta;             // 恒等映射权重，由 theta/layer 计算得出
    private boolean sharedWeights;
    private boolean normalize;

    public GCN2Conv(long channels, float alpha, Float theta, Integer layer,
                    boolean sharedWeights, boolean normalize) {
        super("add");
        this.alpha = alpha;
        this.sharedWeights = sharedWeights;
        this.normalize = normalize;

        // 计算 beta = log(theta / layer + 1)
        if (theta != null && layer != null) {
            this.beta = (float) Math.log(theta / layer + 1.0);
        } else {
            this.beta = 0.1f; // 默认值
        }

        // 严格使用 LinearImpl
        this.lin = new LinearImpl(channels, channels);
        register_module("lin", lin);

        if (!sharedWeights) {
            this.linRes = new LinearImpl(channels, channels);
            register_module("lin_res", linRes);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    
    /**
     * @param x          当前层的特征 [N, channels]
     * @param x_0        初始输入特征 [N, channels]
     * @param edge_index 边索引 [2, E]
     * @param edge_weight 边权重 (可选)
     */
    public Tensor forward(Tensor x, Tensor x_0, Tensor edge_index, Tensor edge_weight) {
        long N = x.size(0);
        Tensor edge_attr = null;
        if (normalize) {
            edge_attr = compute_normalization(edge_index, edge_weight, N);
        } else {
            edge_attr = edge_weight != null ? edge_weight : torch.ones(new long[]{edge_index.size(1)}, x.options());
        }
        // 1. 邻居聚合：\hat{A} @ x
        // 这里需要进行对称归一化处理
//        Tensor norm = null;
//        if (normalize) {
//            norm = compute_normalization(edge_index, edge_weight, N);
//        }

        Tensor out = propagate(edge_index, x, edge_attr);

        // 2. 初始残差连接: (1 - alpha) * \hat{A}x + alpha * x_0
        out = out.mul(new Scalar(1.0 - alpha)).add(x_0.mul(new Scalar(alpha)));

        // 3. 恒等映射与线性变换: (1 - beta) * Identity + beta * W
        if (sharedWeights) {
            // 公式: (1 - beta) * out + beta * W(out)
            out = out.mul(new Scalar(1.0 - beta)).add(lin.forward(out).mul(new Scalar(beta)));
        } else {
            // GCNII* 变体: 区分聚合部分和残差部分的权重
            Tensor out1 = lin.forward(out.mul(new Scalar(1.0 - alpha)));
            Tensor out2 = linRes.forward(x_0.mul(new Scalar(alpha)));
            out = out1.add(out2).mul(new Scalar(beta)).add(out.mul(new Scalar(1.0 - beta)));
        }

        return out;
    }

    private Tensor compute_normalization(Tensor edge_index, Tensor edge_weight, long numNodes) {
        if (edge_weight == null) {
            edge_weight = torch.ones(new long[]{edge_index.size(1)}, edge_index.options());
        }
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);
        Tensor deg = torch.zeros(new long[]{numNodes}, edge_weight.options());
        deg.scatter_add_(0, row, edge_weight);
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));
        return degInvSqrt.index_select(0, row).mul(edge_weight).mul(degInvSqrt.index_select(0, col));
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j.mul(edge_attr.view(-1, 1));
    }
}