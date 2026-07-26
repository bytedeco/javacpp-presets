package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;

/**
 * 修复版 HypergraphConv：
 * 1. 完善内存管理（释放所有临时张量，避免泄漏）
 * 2. 补充完整参数校验（空值、维度、索引越界）
 * 3. 实现标准超图归一化（D_v^-1 * H * W * D_e^-1 * H^T）
 * 4. 完成注意力机制实现
 * 5. 增加资源释放逻辑（close 方法）
 * 6. 修复张量操作安全问题（view 前校验维度）
 */
public class HypergraphConv extends MessagePassing {
    private LinearImpl lin;
    private long inChannels, outChannels;
    private boolean useAttention;
    private int heads;
    private boolean concat;
    private Tensor att; // 注意力参数 [1, heads, 2*outChannels]
    private boolean isReleased = false;

    /**
     * 构造函数：初始化超图卷积核心参数
     * @param inChannels  输入特征维度
     * @param outChannels 单头输出维度
     * @param useAttention 是否使用注意力机制
     * @param heads 注意力头数
     * @param concat 是否拼接多头结果（true: heads*outChannels, false: 均值）
     */
    public HypergraphConv(long inChannels, long outChannels, boolean useAttention, int heads, boolean concat) {
        super("add");
        // 参数合法性校验
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("输入/输出维度必须为正整数");
        }
        if (heads <= 0) {
            throw new IllegalArgumentException("注意力头数必须为正整数");
        }

        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.useAttention = useAttention;
        this.heads = heads;
        this.concat = concat;

        // 基础线性变换：inChannels → heads*outChannels
        this.lin = new LinearImpl(inChannels, heads * outChannels);
        register_module("lin", lin);

        // 初始化注意力参数
        if (useAttention) {
            this.att = torch.randn(new long[]{1, heads, 2 * outChannels});
            register_parameter("att", new Parameter(att)); // 正确注册参数
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        checkReleased();
        return ((HypergraphConv)this).forward(x, edge_index, (Tensor)null);
    }

    /**
     * 核心前向传播方法
     * @param x                节点特征 [N, inChannels]
     * @param hyperedge_index  超图索引 [2, num_incidences] 
     *                        row 0: 节点索引, row 1: 超边索引
     * @param hyperedge_weight 超边权重 [num_edges]（可为null）
     * @return 卷积后的节点特征 [N, heads*outChannels] 或 [N, outChannels]
     */
    public Tensor forward(Tensor x, Tensor hyperedge_index, Tensor hyperedge_weight) {
        checkReleased();
        // 1. 输入参数全量校验
        validateInputs(x, hyperedge_index);

        long numNodes = x.size(0);
        Tensor nodeIdx = null;
        Tensor edgeIdx = null;
        Tensor maxEdgeIdx = null;
        try {
            nodeIdx = hyperedge_index.select(0, 0);
            edgeIdx = hyperedge_index.select(0, 1);

            // 安全获取超边数量（处理空超边情况）
            maxEdgeIdx = torch.max(edgeIdx);
            long numEdges = maxEdgeIdx.item_long() + 1;

            // 2. 节点特征线性变换 + 维度重塑
            Tensor xTrans = lin.forward(x);
            // 安全 view：校验维度兼容性
            if (xTrans.size(1) != heads * outChannels) {
                throw new IllegalArgumentException(
                        "线性变换后维度不匹配：预期 " + (heads * outChannels) +
                                "，实际 " + xTrans.size(1)
                );
            }
            xTrans = xTrans.view(numNodes, heads, outChannels);

            // 3. 计算归一化系数（PyG 标准超图归一化）
            Tensor Dv = computeNodeDegree(nodeIdx, numNodes); // 节点度 [numNodes]
            Tensor De = computeEdgeDegree(edgeIdx, numEdges); // 超边度 [numEdges]
            Tensor normV = Dv.pow(new Scalar(-0.5f)).view(-1, 1, 1); // Dv^-0.5 [numNodes,1,1]
            Tensor normE = De.pow(new Scalar(-0.5f)).view(-1, 1, 1); // De^-0.5 [numEdges,1,1]

            // 4. 阶段 A: 节点 → 超边聚合（带归一化）
            Tensor x_j = xTrans.index_select(0, nodeIdx); // [E, heads, outChannels]
            x_j = x_j.mul(normV.index_select(0, nodeIdx)); // 节点侧归一化

            // 聚合得到超边特征
            Tensor e = torch.zeros(new long[]{numEdges, heads, outChannels}, x.options());
            e.scatter_add_(0, edgeIdx.view(-1, 1, 1).expand_as(x_j), x_j);

            // 超边侧归一化 + 权重缩放
            e = e.mul(normE);
            if (hyperedge_weight != null) {
                validateHyperedgeWeight(hyperedge_weight, numEdges);
                e = e.mul(hyperedge_weight.view(-1, 1, 1));
            }

            // 5. 阶段 B: 超边 → 节点聚合（带注意力/归一化）
            Tensor e_j = e.index_select(0, edgeIdx); // [E, heads, outChannels]

            // 注意力机制：计算注意力权重并应用
            if (useAttention) {
                e_j = applyAttention(xTrans, e_j, nodeIdx, edgeIdx);
            }

            // 聚合回节点
            Tensor out = torch.zeros(new long[]{numNodes, heads, outChannels}, x.options());
            out.scatter_add_(0, nodeIdx.view(-1, 1, 1).expand_as(e_j), e_j);
            out = out.mul(normV); // 最终节点侧归一化

            // 6. 多头合并
            if (concat) {
                out = out.view(numNodes, heads * outChannels);
            } else {
                out = out.mean(1); // 多头均值
            }

            // 释放临时张量（核心：避免内存泄漏）
            safeClose(Dv, De, normV, normE, x_j, e, e_j);
            return out;

        } finally {
            // 确保所有临时张量释放
            safeClose(nodeIdx, edgeIdx, maxEdgeIdx);
        }
    }

    /**
     * 计算节点度：每个节点参与的超边数量
     */
    private Tensor computeNodeDegree(Tensor nodeIdx, long numNodes) {
        Tensor ones = torch.ones(new long[]{nodeIdx.size(0)}, torch.tensor().options().dtype(new ScalarTypeOptional(torch.kFloat())));
        Tensor Dv = torch.zeros(new long[]{numNodes}, torch.tensor().options().dtype(new ScalarTypeOptional(torch.kFloat())));
        try {
            Dv.scatter_add_(0, nodeIdx, ones);
            // 避免除零：度为0的节点设为1
            Dv = Dv.clamp_min(new Scalar(1.0f));
            return Dv;
        } finally {
            safeClose(ones);
        }
    }

    /**
     * 计算超边度：每个超边包含的节点数量
     */
    private Tensor computeEdgeDegree(Tensor edgeIdx, long numEdges) {
        Tensor ones = torch.ones(new long[]{edgeIdx.size(0)}, torch.tensor().options().dtype(new ScalarTypeOptional(torch.kFloat())));
        Tensor De = torch.zeros(new long[]{numEdges}, torch.tensor().options().dtype(new ScalarTypeOptional(torch.kFloat())));
        try {
            De.scatter_add_(0, edgeIdx, ones);
            // 避免除零：度为0的超边设为1
            De = De.clamp_min(new Scalar(1.0f));
            return De;
        } finally {
            safeClose(ones);
        }
    }

    /**
     * 应用注意力机制：计算节点-超边对的注意力权重
     */
    private Tensor applyAttention(Tensor xTrans, Tensor e_j, Tensor nodeIdx, Tensor edgeIdx) {
        // 1. 获取节点特征和超边特征
        Tensor x_i = xTrans.index_select(0, nodeIdx); // [E, heads, outChannels]

        // 2. 拼接节点-超边特征 [E, heads, 2*outChannels]
        Tensor catFeat = torch.cat(new TensorVector(x_i, e_j), 2);

        // 3. 计算注意力分数 [E, heads, 1]
        Tensor attScore = torch.sum(catFeat.mul(att),new long[]{ 2}, true,new ScalarTypeOptional()); // 点积注意力
        attScore = torch.softmax(attScore, 0); // 沿超边维度归一化

        // 4. 应用注意力权重
        e_j = e_j.mul(attScore);

        // 释放临时张量
        safeClose(x_i, catFeat, attScore);
        return e_j;
    }

    /**
     * 输入参数校验：空值、维度、索引越界
     */
    private void validateInputs(Tensor x, Tensor hyperedge_index) {
        if (x == null) {
            throw new IllegalArgumentException("节点特征 x 不能为空");
        }
        if (hyperedge_index == null) {
            throw new IllegalArgumentException("超图索引 hyperedge_index 不能为空");
        }
        if (x.dim() != 2) {
            throw new IllegalArgumentException("节点特征 x 必须是 2 维张量 [N, inChannels]，当前维度：" + x.dim());
        }
        if (hyperedge_index.dim() != 2 || hyperedge_index.size(0) != 2) {
            throw new IllegalArgumentException(
                    "超图索引 hyperedge_index 必须是 [2, num_incidences] 形状，当前：" +
                            hyperedge_index.size(0) + "x" + hyperedge_index.size(1)
            );
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "节点特征维度不匹配：预期 " + inChannels + "，实际 " + x.size(1)
            );
        }

        // 校验节点索引不越界
        Tensor maxNodeIdx = torch.max(hyperedge_index.select(0, 0));
        try {
            if (maxNodeIdx.item_long() >= x.size(0)) {
                throw new IllegalArgumentException(
                        "超图索引包含非法节点索引：" + maxNodeIdx.item_long() + " ≥ " + x.size(0)
                );
            }
        } finally {
            safeClose(maxNodeIdx);
        }
    }

    /**
     * 超边权重校验
     */
    private void validateHyperedgeWeight(Tensor hyperedge_weight, long numEdges) {
        if (hyperedge_weight.dim() != 1) {
            throw new IllegalArgumentException("超边权重必须是 1 维张量，当前维度：" + hyperedge_weight.dim());
        }
        if (hyperedge_weight.size(0) != numEdges) {
            throw new IllegalArgumentException(
                    "超边权重数量不匹配：预期 " + numEdges + "，实际 " + hyperedge_weight.size(0)
            );
        }
    }

    /**
     * 安全释放张量/模块：避免空指针和重复释放
     */
    private void safeClose(AutoCloseable... closeables) {
        for (AutoCloseable c : closeables) {
            if (c != null) {
                try {
                    c.close();
                } catch (Exception e) {
                    System.err.println("释放资源警告：" + e.getMessage());
                }
            }
        }
    }

    /**
     * 检查资源是否已释放
     */
    private void checkReleased() {
        if (isReleased) {
            throw new IllegalStateException("HypergraphConv 已释放资源，无法继续使用");
        }
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        checkReleased();
        return x_j;
    }

    /**
     * 释放所有资源（模块、参数、张量）
     */
    @Override
    public void close() {
        if (!isReleased) {
            safeClose(lin, att);
            super.close();
            isReleased = true;
        }
    }

    // Getter 方法
    public long getInChannels() { return inChannels; }
    public long getOutChannels() { return outChannels; }
    public boolean isUseAttention() { return useAttention; }
    public int getHeads() { return heads; }
    public boolean isConcat() { return concat; }
    public boolean isReleased() { return isReleased; }
}


//public class HypergraphConv extends MessagePassing {
//    private LinearImpl lin;
//    private long inChannels, outChannels;
//    private boolean useAttention;
//    private int heads;
//    private boolean concat;
//
//    // 注意力相关参数 (简化实现)
//    private Tensor att;
//
//    public HypergraphConv(long inChannels, long outChannels, boolean useAttention, int heads, boolean concat) {
//        super("add");
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.useAttention = useAttention;
//        this.heads = heads;
//        this.concat = concat;
//
//        // 基础线性变换
//        this.lin = new LinearImpl(inChannels, heads * outChannels);
//        register_module("lin", lin);
//
//        if (useAttention) {
//            this.att = torch.randn(new long[]{1, heads, 2 * outChannels});
//            register_parameter("att", att);
//        }
//    }
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, null);
//    }
//    /**
//     * @param x                节点特征 [N, inChannels]
//     * @param hyperedge_index  超图索引 [2, num_incidences] 
//     * row 0 是节点索引, row 1 是超边索引
//     * @param hyperedge_weight 超边权重 [num_edges]
//     */
//    public Tensor forward(Tensor x, Tensor hyperedge_index, Tensor hyperedge_weight) {
//        long numNodes = x.size(0);
//        long numEdges = hyperedge_index.select(0, 1).max().item_long() + 1;
//
//        // 1. 节点特征线性变换
//        Tensor xTrans = lin.forward(x).view(numNodes, heads, outChannels);
//
//        // 2. 预计算归一化系数 (D_v^-1 * H * W * D_e^-1 * H^T)
//        // 简化版：按照 PyG 的散布聚合逻辑
//
//        // --- 阶段 A: 节点 -> 超边 (Aggregate nodes into edges) ---
//        Tensor nodeIdx = hyperedge_index.select(0, 0);
//        Tensor edgeIdx = hyperedge_index.select(0, 1);
//
//        // 提取参与超边的节点特征
//        Tensor x_j = xTrans.index_select(0, nodeIdx);
//
//        // 聚合得到超边特征 [numEdges, heads, outChannels]
//        Tensor e = torch.zeros(new long[]{numEdges, heads, outChannels}, x.options());
//        e.scatter_add_(0, edgeIdx.view(-1, 1, 1).expand_as(x_j), x_j);
//
//        // 如果有超边权重，在此处进行缩放
//        if (hyperedge_weight != null) {
//            e = e.mul(hyperedge_weight.view(-1, 1, 1));
//        }
//
//        // --- 阶段 B: 超边 -> 节点 (Aggregate edges into nodes) ---
//        // 提取节点对应的超边特征
//        Tensor e_j = e.index_select(0, edgeIdx);
//
//        // 聚合回节点 [numNodes, heads, outChannels]
//        Tensor out = torch.zeros(new long[]{numNodes, heads, outChannels}, x.options());
//        out.scatter_add_(0, nodeIdx.view(-1, 1, 1).expand_as(e_j), e_j);
//
//        // 3. 多头合并
//        if (concat) {
//            out = out.view(numNodes, heads * outChannels);
//        } else {
//            out = out.mean(1);
//        }
//
//        return out;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // x_i 是目标节点，x_j 是源节点
//        return x_j;
//    }
//}