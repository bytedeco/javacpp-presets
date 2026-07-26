package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 有向图 GNN 卷积层（DirGNNConv）
 * 核心逻辑：分别聚合入边（j→i）和出边（i→j）信息，通过 alpha 加权融合，支持根节点残差
 * 继承 MessagePassing 基类（图卷积层标准规范）
 */
public class DirGNNConv extends MessagePassing {
    private MessagePassing conv;       // 基础消息传递算子（如 SAGEConvV2）
    private float alpha;               // 入边权重系数（出边权重=1-alpha），范围 [0,1]
    private boolean rootWeight;        // 是否包含根节点（自连接）权重
    private LinearImpl linRoot;        // 根节点特征变换层
    private long inChannels;           // 输入通道数
    private long outChannels;          // 输出通道数

    /**
     * 构造方法：初始化有向图卷积层
     * @param conv 基础消息传递算子（如 SAGEConvV2）
     * @param alpha 入边权重系数（0≤alpha≤1）
     * @param rootWeight 是否启用根节点变换
     * @param inChannels 输入通道数
     * @param outChannels 输出通道数
     */
    public DirGNNConv(MessagePassing conv, float alpha, boolean rootWeight, long inChannels, long outChannels) {
//        super(false); // 初始化 MessagePassing 基类（稠密/稀疏模式：false=稠密）
        // 校验 alpha 范围
        if (alpha < 0.0f || alpha > 1.0f) {
            throw new IllegalArgumentException("alpha 必须在 [0,1] 范围内，当前值：" + alpha);
        }
        // 校验基础卷积算子非空
        if (conv == null) {
            throw new IllegalArgumentException("基础卷积算子 conv 不能为空");
        }

        this.conv = conv;
        this.alpha = alpha;
        this.rootWeight = rootWeight;
        this.inChannels = inChannels;
        this.outChannels = outChannels;

        // 1. 注册基础卷积算子到模块树
        register_module("conv", conv);

        // 2. 初始化并注册根节点变换层
        if (rootWeight) {
            this.linRoot = new LinearImpl(inChannels, outChannels);
            register_module("lin_root", linRoot);
        }
    }

    /**
     * 前向传播：有向图消息聚合
     * @param x 节点特征 [N, inChannels] 或 [B, N, inChannels]
     * @param edge_index 有向边索引 [2, E]（第一行=源节点，第二行=目标节点）
     * @return 聚合后节点特征 [N, outChannels] 或 [B, N, outChannels]
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // ========== 输入校验 ==========
        // 1. 特征张量维度校验
        if (x.dim() != 2 && x.dim() != 3) {
            throw new IllegalArgumentException("x 必须是 2 维 [N,C] 或 3 维 [B,N,C] 张量，当前维度：" + x.dim());
        }
        // 2. 边索引维度校验
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index 必须是 [2, E] 张量，当前维度：" + edge_index.dim() + "，第一维大小：" + edge_index.size(0));
        }
        // 3. 特征通道数校验
        long xChannels = (x.dim() == 2) ? x.size(1) : x.size(2);
        if (xChannels != this.inChannels) {
            throw new IllegalArgumentException("x 通道数不匹配：期望 " + inChannels + "，实际 " + xChannels);
        }

        // ========== 设备/类型对齐 ==========
        // 保证基础卷积算子、根节点层与输入 x 设备/类型一致
        this.conv.to(x.device(), x.scalar_type(),false);
        if (this.linRoot != null) {
            this.linRoot.to(x.device(), x.scalar_type(),false);
        }

        // ========== 有向图消息聚合核心逻辑 ==========
        // 1. 入边聚合（j→i）：使用原始 edge_index
        Tensor outIn = this.conv.forward(x, edge_index);

        // 2. 出边聚合（i→j）：翻转 edge_index 的两行（源/目标节点互换）
        Tensor edgeRow0 = edge_index.select(0, 0);  // 源节点行
        Tensor edgeRow1 = edge_index.select(0, 1);  // 目标节点行
        Tensor revEdgeIndex = torch.stack(new TensorVector(edgeRow1, edgeRow0), 0); // 翻转后：目标→源（出边）
        Tensor outOut = this.conv.forward(x, revEdgeIndex);

        // 3. 凸组合融合入边/出边特征：alpha*入边 + (1-alpha)*出边
        Tensor out = outIn.mul(new Scalar(this.alpha))
                .add(outOut.mul(new Scalar(1.0f - this.alpha)));

        // 4. 根节点特征残差连接（自连接）
        if (this.rootWeight && this.linRoot != null) {
            Tensor rootOut = this.linRoot.forward(x);
            out = out.add(rootOut);
            rootOut.close(); // 释放临时张量
        }

        // ========== 释放临时张量 ==========
        edgeRow0.close();
        edgeRow1.close();
        revEdgeIndex.close();
        outIn.close();
        outOut.close();

        return out;
    }

    /**
     * 重置所有可训练参数（必须复写基类方法）
     */
//    @Override
    public void reset_parameters() {
        // 重置基础卷积算子参数
//        if (this.conv != null) {
//            this.conv.reset_parameters();
//        }
        // 重置根节点变换层参数
        if (this.linRoot != null) {
            this.linRoot.reset_parameters();
        }
    }

    /**
     * 复写 message 方法（MessagePassing 基类要求）
     * 有向图聚合逻辑已在 forward 中实现，此处仅兼容签名
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j; // 基础 message 逻辑：返回邻居特征
    }

    // ====== 辅助方法：获取配置/参数（用于测试验证） ======
    public MessagePassing getConv() {
        return this.conv;
    }

    public float getAlpha() {
        return this.alpha;
    }

    public boolean isRootWeight() {
        return this.rootWeight;
    }

    public LinearImpl getLinRoot() {
        return this.linRoot;
    }

    // ====== 资源释放：避免JNI内存泄漏 ======
    @Override
    public void close() {
        // 释放基础卷积算子
        if (this.conv != null) {
            this.conv.close();
            this.conv = null;
        }
        // 释放根节点变换层
        if (this.linRoot != null) {
            this.linRoot.close();
            this.linRoot = null;
        }
        // 释放基类资源
        super.close();
    }

    // 防止GC时未手动close导致内存泄漏
//    @Override
//    protected void finalize() throws Throwable {
//        try {
//            close();
//        } finally {
//            super.finalize();
//        }
//    }
}

// 补充：SAGEConvV2 实现（适配 DirGNNConv 依赖）
//class SAGEConvV2 extends MessagePassing {
//    private LinearImpl lin;
//    private long inChannels;
//    private long outChannels;
//
//    public SAGEConvV2(long inChannels, long outChannels) {
////        super(false);
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.lin = new LinearImpl(inChannels, outChannels);
//        register_module("lin", lin);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        // 简化版 SAGE 聚合：均值聚合邻居特征 + 线性变换
//        Tensor out = aggregate(x, edge_index); // 均值聚合
//        return lin.forward(out);
//    }
//
//    private Tensor aggregate(Tensor x, Tensor edge_index) {
//        // 模拟均值聚合：此处简化实现（适配测试）
//        long N = x.size(0);
//        Tensor adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), new long[]{N, N});
//        Tensor deg = adj.sum(new long[]{1}, true, new ScalarTypeOptional(torch.kFloat()));
//        Tensor out = adj.matmul(x).div(deg.add(new Scalar(1e-6)));
//        adj.close();
//        deg.close();
//        return out;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        return x_j;
//    }
//
////    @Override
//    public void reset_parameters() {
//        if (lin != null) lin.reset_parameters();
//    }
//
//    @Override
//    public void close() {
//        if (lin != null) lin.close();
//        super.close();
//    }
//}

//package org.bytedeco.pytorch.geometric.nn.conv;
//


//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.nn.Module;
//import org.bytedeco.pytorch.global.torch;
//
///**
// * 严格使用 LinearImpl 规范实现 torch_geometric.nn.conv.DirGNNConv
// * 针对有向图优化，分别聚合入边和出边信息。
// */
//public class DirGNNConv extends Module {
//    private MessagePassing conv;    // 基础消息传递算子
//    private float alpha;            // 入边和出边的权重分配系数
//    private boolean rootWeight;     // 是否包含根节点（自连接）权重
//
//    // 严格使用 LinearImpl 处理根节点变换
//    private LinearImpl linRoot;
//
//    public DirGNNConv(SAGEConvV2 conv, float alpha, boolean rootWeight, long inChannels, long outChannels) {
//        super();
//        this.conv = conv;
//        this.alpha = alpha;
//        this.rootWeight = rootWeight;
//
//        // 1. 注册基础卷积算子
//        register_module("conv", conv);
//
//        // 2. 根节点变换注册
//        if (rootWeight) {
//            this.linRoot = new LinearImpl(inChannels, outChannels);
//            register_module("lin_root", linRoot);
//        }
//    }
//
//    /**
//     * @param x          节点特征 [N, inChannels]
//     * @param edge_index 有向边索引 [2, E]
//     */
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        // 1. 计算入边消息 (In-edges: j -> i)
//        // 默认的 edge_index 就是入边方向
//        Tensor outIn = conv.forward(x, edge_index);
//
//        // 2. 计算出边消息 (Out-edges: i -> j)
//        // 通过翻转 edge_index 的两行来实现反向聚合
//        Tensor revEdgeIndex = torch.stack(new TensorVector(edge_index.select(0, 1), edge_index.select(0, 0)), 0);
//        Tensor outOut = conv.forward(x, revEdgeIndex);
//
//        // 3. 凸组合: alpha * Out_in + (1 - alpha) * Out_out
//        Tensor out = outIn.mul(new Scalar(alpha)).add(outOut.mul(new Scalar(1.0f - alpha)));
//
//        // 4. 加上根节点特征 (Root Transformation)
//        if (rootWeight && linRoot != null) {
//            out = out.add(linRoot.forward(x));
//        }
//
//        return out;
//    }
//
////    @Override
//    public void reset_parameters() {
//        if (conv != null) conv.asSequential().reset();//.reset_parameters();
//        if (linRoot != null) linRoot.reset_parameters();
//    }
//}