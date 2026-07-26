package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 torch_geometric.nn.conv.XConv (PointCNN)
 * 特点：学习 X-变换矩阵以实现点云上的有序卷积。
 */

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.global.torch.*;

public class XConv extends Module {
    private int dim, kernelSize, dilation;
    private long inChannels, outChannels;

    private SequentialImpl mlpPos; // 使用 SequentialImpl 替代 Module 接口以便调用 forward
    private SequentialImpl mlpX;
    private Tensor weight;
    private Tensor bias;

    public XConv(long inChannels, long outChannels, int dim, int kernelSize, Integer hiddenChannels, int dilation, boolean hasBias) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.dim = dim;
        this.kernelSize = kernelSize;
        this.dilation = dilation;

        if (hiddenChannels == null) hiddenChannels = (int) (inChannels / 4);
        if (hiddenChannels <= 0) hiddenChannels = 8; // 兜底

        // 1. 修正：创建具体的 MLP 实现
        this.mlpPos = createPosMLP(dim, hiddenChannels);
        this.mlpX = createXMLP(kernelSize, dim);

        // 2. 卷积权重: [K, C_in + C_hidden, C_out]
        this.weight = torch.randn(new long[]{kernelSize, hiddenChannels + (int)inChannels, (int)outChannels});
        torch.xavier_uniform_(this.weight);

        // 3. 注册组件
        register_module("mlp_pos", mlpPos);
        register_module("mlp_x", mlpX);
        register_parameter("weight", weight);

        if (hasBias) {
            this.bias = torch.zeros(new long[]{outChannels});
            register_parameter("bias", bias);
        }
    }

    public Tensor forward(Tensor x, Tensor pos, Tensor batch) {
        long N = pos.size(0);
        TensorOptions floatOpt = x.options();

        // 1. k-NN 获取邻域点
        Tensor knnIdx = computeKNN(pos, kernelSize, dilation, batch); // [N, K]

        // 2. 坐标中心化: pos_j - pos_i
        Tensor targetPos = pos.unsqueeze(1); // [N, 1, dim]
        Tensor neighborPos = pos.index_select(0, knnIdx.reshape(-1)).view(N, kernelSize, dim);
        Tensor relPos = neighborPos.sub(targetPos); // [N, K, dim]

        // 3. 提升坐标维度 (Lifting)
        // [N*K, dim] -> [N*K, hidden] -> [N, K, hidden]
        Tensor liftedPos = mlpPos.forward(relPos.reshape(-1, dim)).view(N, kernelSize, -1);

        // 4. 拼接原始特征
        Tensor neighborX = x.index_select(0, knnIdx.reshape(-1)).view(N, kernelSize, inChannels);
        Tensor f = torch.cat(new TensorVector(liftedPos, neighborX), -1); // [N, K, C_comb]

        // 5. 学习 X-变换矩阵
        // 输入 [N, K*dim], 输出 [N, K*K] -> [N, K, K]
        Tensor transformMat = mlpX.forward(relPos.reshape(N, kernelSize * dim)).view(N, kernelSize, kernelSize);

        // 6. 应用 X-变换: X @ f
        Tensor f_transformed = torch.matmul(transformMat, f); // [N, K, C_comb]

        // 7. 深度卷积模拟
        // 我们利用 einsum 实现 [N, K, C_comb] 与 [K, C_comb, C_out] 的高效收缩
        // 逻辑等价于对每个 K 维度应用特定的 weight 片段
        Tensor out = torch.einsum("nkc, kcd -> nd", new TensorVector(f_transformed, weight));

        if (bias != null) out = out.add(bias);
        return out;
    }

    private SequentialImpl createPosMLP(int dim, int hidden) {
        SequentialImpl sequential = new SequentialImpl();
        sequential.push_back(new LinearImpl(dim, hidden / 2));
        sequential.push_back(new ReLUImpl());
        sequential.push_back(new LinearImpl(hidden / 2, hidden));
        sequential.push_back(new ReLUImpl());
        return sequential;
//        return new SequentialImpl(
//                new LinearImpl(dim, hidden / 2),
//                new ReLUImpl(),
//                new LinearImpl(hidden / 2, hidden),
//                new ReLUImpl()
//        );
    }

    private SequentialImpl createXMLP(int k, int dim) {
        SequentialImpl sequential = new SequentialImpl();
        sequential.push_back(new LinearImpl(k * dim, k * k));
        sequential.push_back(new ReLUImpl());
        sequential.push_back(new LinearImpl(k * k, k * k));
        sequential.push_back(new ReLUImpl());
        return sequential;
    }

    private Tensor computeKNN(Tensor pos, int k, int dilation, Tensor batch) {
        // 模拟外部 knn 算子返回 [N, K]
        return torch.randint(0, pos.size(0), new long[]{pos.size(0), k},
                new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
    }
}
//public class XConv extends Module {
//    private int dim, kernelSize, dilation;
//    private long inChannels, outChannels;
//
//    private Module mlpPos;     // h_delta: 将相对位置提升到高维空间
//    private Module mlpX;       // h_X: 学习 X-变换矩阵 [K, K]
//    private Tensor weight;     // 卷积核权重 [K, hidden_channels + in_channels, outChannels]
//    private Tensor bias;
//
//    public XConv(long inChannels, long outChannels, int dim, int kernelSize, Integer hiddenChannels, int dilation, boolean hasBias) {
//        super();
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.dim = dim;
//        this.kernelSize = kernelSize;
//        this.dilation = dilation;
//
//        if (hiddenChannels == null) hiddenChannels = (int) (inChannels / 4);
//
//        // 1. MLP_delta: [dim] -> [hidden_channels]
//        // 通常定义为两层 LinearImpl
//        this.mlpPos = createPosMLP(dim, hiddenChannels);
//
//        // 2. MLP_X: [K, dim] -> [K, K]
//        // 用于学习点云的排列转换矩阵
//        this.mlpX = createXMLP(kernelSize, dim);
//
//        // 3. 卷积权重
//        this.weight = torch.randn(new long[]{kernelSize, hiddenChannels + inChannels, outChannels});
//        torch.xavier_uniform_(this.weight);
//
//        register_module("mlp_pos", mlpPos);
//        register_module("mlp_x", mlpX);
//        register_parameter("weight", weight);
//
//        if (hasBias) {
//            this.bias = torch.zeros(new long[]{outChannels});
//            register_parameter("bias", bias);
//        }
//    }
//
//    /**
//     * @param x   节点特征 [N, inChannels]
//     * @param pos 节点坐标 [N, dim]
//     * @param batch 批次向量
//     */
//    public Tensor forward(Tensor x, Tensor pos, Tensor batch) {
//        long N = pos.size(0);
//
//        // 1. k-NN 获取邻域点 (考虑 dilation)
//        // 这里的 k 对应 kernelSize
//        Tensor knnIdx = computeKNN(pos, kernelSize, dilation, batch); // [N, K]
//
//        // 2. 坐标中心化: pos_j - pos_i
//        Tensor targetPos = pos.unsqueeze(1); // [N, 1, dim]
//        Tensor neighborPos = pos.index_select(0, knnIdx.view(-1)).view(N, kernelSize, dim);
//        Tensor relPos = neighborPos.sub(targetPos); // [N, K, dim]
//
//        // 3. 提升坐标维度 (Lifting)
//        Tensor liftedPos = mlpPos.asSequential().forward(relPos.view(-1, dim)).view(N, kernelSize, -1);
//
//        // 4. 拼接原始特征
//        Tensor neighborX = x.index_select(0, knnIdx.view(-1)).view(N, kernelSize, inChannels);
//        Tensor f = torch.cat(new TensorVector(liftedPos, neighborX), -1); // [N, K, C_lifted + C_in]
//
//        // 5. 学习 X-变换矩阵
//        // mlpX 输入相对坐标，输出 K*K 矩阵
//        Tensor transformMat = mlpX.asSequential().forward(relPos.view(-1, kernelSize * dim)).view(N, kernelSize, kernelSize);
//
//        // 6. 应用 X-变换: X @ f
//        Tensor f_transformed = torch.matmul(transformMat, f); // [N, K, C_combined]
//
//        // 7. 深度可分离卷积 (模拟标准卷积)
//        // [N, K, C] -> 对 K 维度应用 weight
//        Tensor out = torch.matmul(f_transformed.transpose(1, 2), weight); // [N, C_combined, outChannels]
//
//        // 聚合 K 维度 (PointCNN 特有的卷积聚合)
//        out = out.sum(1);
//
//        if (bias != null) out = out.add(bias);
//        return out;
//    }
//
//    // 辅助方法：创建位置提升 MLP
//    private Module createPosMLP(int dim, int hidden) {
//        // 实现建议: Linear(dim, hidden/2) -> ReLU -> Linear(hidden/2, hidden)
//        return null; // 需传入具体的 Sequential
//    }
//
//    // 辅助方法：创建 X 矩阵学习 MLP
//    private Module createXMLP(int k, int dim) {
//        // 实现建议: Linear(k*dim, k*k) -> ...
//        return null;
//    }
//
//    private Tensor computeKNN(Tensor pos, int k, int dilation, Tensor batch) {
//        // 模拟外部 knn 算子
//        return torch.zeros(new long[]{pos.size(0), k}, new TensorOptions().dtype(new ScalarTypeOptional(torch.kLong())));
//    }
//}