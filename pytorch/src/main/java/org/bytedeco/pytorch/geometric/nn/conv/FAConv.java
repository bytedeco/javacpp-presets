package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 修正版 FAConv (频率自适应图卷积)
 * 核心功能：捕获图信号的低通和高通分量，支持注意力系数+归一化+残差连接
 * 修复点：
 * 1. 修正 forward 方法重载和参数传递
 * 2. 替换原地操作，避免叶子张量报错
 * 3. 完善消息传递逻辑，整合 alpha 和 norm
 * 4. 增加输入校验和边界处理
 * 5. 补充 dropout 功能
 */
public class FAConv extends MessagePassing {
    public LinearImpl lin; // 注意力系数计算的线性层
    private float eps;      // 初始残差权重
    private float dropout;  // dropout 概率
    private boolean normalize; // 是否启用对称归一化
    private Tensor alpha;   // 临时存储注意力系数（供 message 方法使用）
    private Tensor norm;    // 临时存储归一化系数（供 message 方法使用）

    // 核心构造函数（匹配 torch_geometric 官方接口）
    public FAConv(long channels, float eps, float dropout, boolean normalize) {
        super("add");
        this.eps = eps;
        this.dropout = dropout;
        this.normalize = normalize;

        // 初始化线性层（输出维度1，用于计算注意力系数）
        this.lin = new LinearImpl(channels, 1);
        register_module("lin", lin);
    }

    // 基础 forward 重载（无初始残差+无边权重）
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, x, edge_index, null); // 无初始残差时，x_0 = x
    }

    // 完整 forward 方法（匹配官方接口）
    public Tensor forward(Tensor x, Tensor x_0, Tensor edge_index, Tensor edge_weight) {
        // ========== 1. 输入校验 ==========
        long[] xShape = x.sizes().vec().get();
        if (xShape.length != 2) {
            throw new IllegalArgumentException("节点特征 x 必须是 2D 张量，当前维度：" + xShape.length);
        }
        long N = xShape[0]; // 节点数
        long channels = xShape[1]; // 特征维度

        long[] edgeIndexShape = edge_index.sizes().vec().get();
        if (edgeIndexShape.length != 2 || edgeIndexShape[0] != 2) {
            throw new IllegalArgumentException("边索引 edge_index 必须是 [2, E] 形状，当前：" + edgeIndexShape);
        }

        // ========== 2. 计算归一化系数 ==========
        this.norm = null;
        if (normalize) {
            this.norm = compute_normalization(edge_index, edge_weight, N);
        }

        // ========== 3. 计算自适应注意力系数 alpha ==========
        // alpha_ij = tanh(lin(x_i) + lin(x_j)) [E, 1]
        Tensor h = lin.forward(x); // [N, 1]
        Tensor sourceIdx = edge_index.select(0, 0); // 源节点索引 [E]
        Tensor targetIdx = edge_index.select(0, 1); // 目标节点索引 [E]

        Tensor h_i = h.index_select(0, sourceIdx); // [E, 1]
        Tensor h_j = h.index_select(0, targetIdx); // [E, 1]
        this.alpha = torch.tanh(h_i.add(h_j));     // [E, 1]

        // 应用 dropout（训练场景）
        if (dropout > 0 ) {
            this.alpha = torch.dropout(this.alpha, dropout, true);
        }

        // ========== 4. 消息传递核心逻辑 ==========
        Tensor out = propagate(edge_index, x, new long[]{N, N}); // [N, channels]

        // ========== 5. 合并初始残差（Initial Residual） ==========
        Tensor residual = x_0.mul(new Scalar(eps)); // eps * x_0
        out = out.add(residual);

        // ========== 6. 资源清理 ==========
        h.close();
        h_i.close();
        h_j.close();
        sourceIdx.close();
        targetIdx.close();
        residual.close();

        return out;
    }

    // 对称归一化计算（替换原地操作，避免叶子张量报错）
    private Tensor compute_normalization(Tensor edge_index, Tensor edge_weight, long numNodes) {
        // 初始化边权重（无边权重时设为1）
        if (edge_weight == null) {
            edge_weight = torch.ones(new long[]{edge_index.size(1)},
                    new TensorOptions().dtype(new ScalarTypeOptional(xdtype(edge_index))).device(new DeviceOptional(xdevice(edge_index))));
        }

        Tensor row = edge_index.select(0, 0); // 源节点 [E]
        Tensor deg = torch.zeros(new long[]{numNodes}, edge_weight.options());
        // 替换原地操作 scatter_add_ → 非原地 scatter_add
        deg = deg.scatter_add(0, row, edge_weight); // 度统计 [N]

        // 计算度的逆平方根（避免除零）
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        // 替换原地操作 masked_fill_ → 非原地 masked_fill
        Scalar zero = new Scalar(0);
        degInvSqrt = degInvSqrt.masked_fill(degInvSqrt.isinf(), zero);

        // 对称归一化: norm_ij = D_i^-0.5 * w_ij * D_j^-0.5 [E]
        Tensor degInvSqrtRow = degInvSqrt.index_select(0, row);
        Tensor col = edge_index.select(0, 1);
        Tensor degInvSqrtCol = degInvSqrt.index_select(0, col);

        Tensor norm = degInvSqrtRow.mul(edge_weight).mul(degInvSqrtCol);

        // 资源清理
        row.close();
        col.close();
        deg.close();
        degInvSqrt.close();
        degInvSqrtRow.close();
        degInvSqrtCol.close();

        return norm;
    }

    // ========== 核心：消息传递逻辑 ==========
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_j: 邻居节点特征 [E, channels]
        // alpha: 注意力系数 [E, 1]
        // norm: 归一化系数 [E]

        Tensor msg = x_j; // 初始消息 = 邻居特征

        // 1. 应用归一化系数
        if (norm != null) {
            // 扩展 norm 维度匹配 x_j: [E] → [E, 1]
            Tensor normExpanded = norm.unsqueeze(1);
            msg = msg.mul(normExpanded);
            normExpanded.close();
        }

        // 2. 应用注意力系数 alpha
        if (alpha != null) {
            msg = msg.mul(alpha); // [E, channels] * [E, 1] = [E, channels]
        }

        return msg;
    }

    // ========== 工具方法：获取张量 dtype/device ==========
    private torch.ScalarType xdtype(Tensor t) {
        return t.options().dtype().toScalarType();
    }

    private Device xdevice(Tensor t) {
        return t.options().device();
    }

    // ========== 资源释放 ==========
//    @Override
//    protected void finalize() throws Throwable {
//        try {
//            if (lin != null) lin.close();
//            if (alpha != null) alpha.close();
//            if (norm != null) norm.close();
//        } finally {
//            super.finalize();
//        }
//    }
}

//package org.bytedeco.pytorch.geometric.nn.conv;
//
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.global.torch;
//
///**
// * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.FAConv
// * 频率自适应图卷积，能够同时捕获图信号的低通和高通分量。
// */
//public class FAConv extends MessagePassing {
//    public LinearImpl lin; // 用于计算注意力系数 alpha 的参数网络
//    private float eps;
//    private float dropout;
//    private boolean normalize;
//
//    public FAConv(long channels, float eps, float dropout, boolean normalize) {
//        super("add");
//        this.eps = eps;
//        this.dropout = dropout;
//        this.normalize = normalize;
//
//        // 严格使用 LinearImpl
//        // 注意：FAConv 的注意力计算通常是基于源和目标特征的线性映射
//        this.lin = new LinearImpl(channels, 1);
//        register_module("lin", lin);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, null);
//    }
//    /**
//     * @param x          当前层的节点特征 [N, channels]
//     * @param x_0        初始输入特征 [N, channels] (Initial Residual)
//     * @param edge_index 边索引 [2, E]
//     */
//    public Tensor forward(Tensor x, Tensor x_0, Tensor edge_index, Tensor edge_weight) {
//        long N = x.size(0);
//
//        // 1. 计算归一化系数
//        Tensor norm = null;
//        if (normalize) {
//            norm = compute_normalization(edge_index, edge_weight, N);
//        }
//
//        // 2. 计算自适应注意力系数 alpha
//        // alpha_ij = tanh(lin(x_i) + lin(x_j))
//        Tensor h = lin.forward(x); // [N, 1]
//        Tensor sourceIdx = edge_index.select(0, 0);
//        Tensor targetIdx = edge_index.select(0, 1);
//
//        Tensor alpha = h.index_select(0, sourceIdx).add(h.index_select(0, targetIdx));
//        alpha = torch.tanh(alpha);
//
//        // 3. 执行消息传递: x_i' = eps * x_0 + sum(alpha_ij * norm_ij * x_j)
//        // 注意：这里 alpha 改变了邻域聚合的方向和强度
//        Tensor out = propagate(edge_index, x, norm, alpha);
//
//        // 4. 合并初始残差
//        return out.add(x_0.mul(new Scalar(eps)));
//    }
//
//    private Tensor compute_normalization(Tensor edge_index, Tensor edge_weight, long numNodes) {
//        if (edge_weight == null) {
//            edge_weight = torch.ones(new long[]{edge_index.size(1)}, edge_index.options());
//        }
//        Tensor row = edge_index.select(0, 0);
//        Tensor col = edge_index.select(0, 1);
//        Tensor deg = torch.zeros(new long[]{numNodes}, edge_weight.options());
//        deg.scatter_add_(0, row, edge_weight);
//
//        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
//        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));
//
//        // 对称归一化: D^-0.5 * A * D^-0.5
//        return degInvSqrt.index_select(0, row).mul(edge_weight).mul(degInvSqrt.index_select(0, col));
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 在这里，edge_attr 传入的是预计算的 norm
//        // 我们需要额外应用 alpha (注意力)
//        // 注意：在复杂的 JavaCPP 调用中，alpha 需要通过 Tensor 传递给 message
//        return x_j; // 简化展示，实际需处理 alpha 乘积
//    }
//}