package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * 严格对齐PyTorch Geometric官方SAGEConv实现
 * 支持：
 * 1. in_channels: int（普通图）/ long[2]（二部图，源/目标维度不同）
 * 2. root_weight: 是否启用自身特征
 * 3. project: 聚合前投影
 * 4. 自定义聚合方式（mean/max/lstm等）
 * 5. L2归一化、偏置等官方参数
 */
public class SAGEConvV3 extends MessagePassing {

    // 核心参数（对齐官方）
    private final long[] inChannels;       // [inSrc, inDst] 或 [inDim, inDim]
    private final long outChannels;
    private final String aggr;             // 聚合方式：mean/max/lstm
    private final boolean normalize;
    private final boolean rootWeight;
    private final boolean project;
    private final boolean bias;

    // 线性层（按需初始化）
    private LinearImpl linProj;            // project=True时的聚合前投影层
    private LinearImpl linNeighbor;        // 邻居特征变换层
    private LinearImpl linRoot;            // 自身特征变换层（root_weight=True）

    // ========== 构造函数（对齐官方API） ==========
    /**
     * 完整构造函数（对齐PyG官方）
     * @param inChannels 输入维度：int（普通图）/ long[2]（二部图）
     * @param outChannels 输出维度
     * @param aggr 聚合方式（默认mean）
     * @param normalize 是否L2归一化
     * @param rootWeight 是否启用自身特征
     * @param project 是否聚合前投影

     */
    public SAGEConvV3(long inChannels, long outChannels, String aggr, boolean normalize,
                    boolean rootWeight, boolean project, boolean bias) {
        this(new long[]{inChannels, inChannels}, outChannels, aggr, normalize, rootWeight, project, bias);
    }

    /**
     * 二部图构造函数（inChannels为[src, dst]）
     */
    public SAGEConvV3(long[] inChannels, long outChannels, String aggr, boolean normalize,
                    boolean rootWeight, boolean project, boolean bias) {
        super(aggr != null ? aggr : "mean"); // 聚合方式传递给MessagePassing基类
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.aggr = aggr != null ? aggr : "mean";
        this.normalize = normalize;
        this.rootWeight = rootWeight;
        this.project = project;
        this.bias = bias;

        // 1. 初始化聚合前投影层（project=True）
        if (this.project) {
            var option = new LinearOptions(this.inChannels[0], this.inChannels[0]);
            option.bias().put(bias);
            this.linProj = register_module("linProj",
                    new LinearImpl(option));
        }

        // 2. 初始化邻居特征变换层
        long inNeighborDim = this.inChannels[0];
        if (this.project) inNeighborDim = this.inChannels[0]; // project后维度不变
        var option = new LinearOptions(inNeighborDim, outChannels);
        option.bias().put(bias);
        this.linNeighbor = register_module("linNeighbor",
                new LinearImpl(option));

        // 3. 初始化自身特征变换层（root_weight=True）
        if (this.rootWeight) {
            long inRootDim = this.inChannels[1];
            var option2 = new LinearOptions(inRootDim, outChannels);
            option2.bias().put(bias);
            this.linRoot = register_module("linRoot",
                    new LinearImpl(option2));
        }
    }

    /**
     * 简化构造函数（默认参数：mean聚合、不归一化、启用自身特征、不投影、使用偏置）
     */
    public SAGEConvV3(long inChannels, long outChannels) {
        this(inChannels, outChannels, "mean", false, true, false, true);
    }

    /**
     * 二部图简化构造函数
     */
    public SAGEConvV3(long[] inChannels, long outChannels) {
        this(inChannels, outChannels, "mean", false, true, false, true);
    }

    // ========== 核心前向传播（对齐官方逻辑） ==========
    /**
     * 普通图前向传播（单节点特征）
     * @param x 节点特征 [N, inDim]
     * @param edgeIndex 边索引 [2, E]
     * @return 输出特征 [N, outDim]
     */
    public Tensor forward(Tensor x, Tensor edgeIndex) {
        return forward(x, x, edgeIndex); // 普通图：src=dst=x
    }

    /**
     * 二部图前向传播（源/目标节点特征分离）
     * @param xSrc 源节点特征 [N_src, inSrc]
     * @param xDst 目标节点特征 [N_dst, inDst]
     * @param edgeIndex 边索引 [2, E] (src→dst)
     * @return 目标节点输出特征 [N_dst, outDim]
     */
    public Tensor forward(Tensor xSrc, Tensor xDst, Tensor edgeIndex) {
        long N_dst = xDst.size(0);
        long[] size = new long[]{xSrc.size(0), N_dst};

        // Step 1: 聚合前投影（project=True）
        if (this.project) {
            xSrc = relu(this.linProj.forward(xSrc));
        }

        // Step 2: 消息传递 + 聚合邻居特征
        Tensor aggrOut = propagate(edgeIndex, xSrc, size);

        // Step 3: 变换邻居特征
        Tensor out = this.linNeighbor.forward(aggrOut);

        // Step 4: 融合自身特征（root_weight=True）
        if (this.rootWeight) {
            Tensor rootOut = this.linRoot.forward(xDst);
            out = out.add(rootOut);
        }

        // Step 5: 激活 + 归一化（GraphSAGE标准流程）
        out = relu(out);
        if (this.normalize) {
            Tensor norm = out.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);
            norm = norm.clamp_min(new Scalar(1e-12)); // 防止除0
            out = out.div(norm);
        }

        return out;
    }

    // ========== MessagePassing 基类方法 ==========
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edgeIndex, Tensor edgeAttr, long numNodes) {
        // GraphSAGE的message就是邻居特征本身（x_j）
        return x_j;
    }

    // ========== 工具方法（获取层参数，方便测试/调试） ==========
    public LinearImpl getLinProj() {
        return linProj;
    }

    public LinearImpl getLinNeighbor() {
        return linNeighbor;
    }

    public LinearImpl getLinRoot() {
        return linRoot;
    }

    public long[] getInChannels() {
        return inChannels;
    }

    public boolean isRootWeight() {
        return rootWeight;
    }

    public boolean isProject() {
        return project;
    }
}