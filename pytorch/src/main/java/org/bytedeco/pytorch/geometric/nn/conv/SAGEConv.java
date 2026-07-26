package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.*;

import static org.bytedeco.pytorch.global.torch.relu;
//import org.gnn.framework.nn.org.bytedeco.pytorch.geometric.nn.conv.MessagePassing;


import static org.bytedeco.pytorch.global.torch.*;

/**
 * 修复后的 GraphSAGE 卷积层实现
 * 遵循标准 GraphSAGE 逻辑：Mean Aggregation + 自身特征融合 + 激活 + L2归一化
 */
public class SAGEConv extends MessagePassing {

    private final LinearImpl linNeighbor; // 邻居特征变换层 (W_neighbor)
    private final LinearImpl linSelf;     // 自身特征变换层 (W_self)
    private final boolean normalize;      // 是否启用L2归一化
    private final boolean bias;           // 是否使用偏置

    // ========== 构造函数（简化+规范） ==========
    /**
     * 标准构造函数（普通图场景）
     * @param inDim  输入特征维度
     * @param outDim 输出特征维度
     * @param normalize 是否启用L2归一化
     * @param bias 是否使用偏置
     */
    public SAGEConv(long inDim, long outDim, boolean normalize, boolean bias) {
        super("mean"); // 固定使用mean聚合（GraphSAGE默认）
        this.normalize = normalize;
        this.bias = bias;

        var option = new LinearOptions(inDim, outDim);
        option.bias().put(bias);
        // 初始化线性层：邻居和自身特征都映射到outDim维度
        this.linNeighbor = register_module("linNeighbor", new LinearImpl(option));
        this.linSelf = register_module("linSelf", new LinearImpl(option));
    }

    /**
     * 简化构造函数（默认：不归一化，使用偏置）
     */
    public SAGEConv(long inDim, long outDim) {
        this(inDim, outDim, false, true);
    }

    // ========== 普通图前向传播（核心方法） ==========
    /**
     * 普通图前向传播（单节点特征，最常用）
     * @param x 节点特征 [N, inDim]
     * @param edge_index 边索引 [2, E]
     * @return 输出特征 [N, outDim]
     */
    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. 聚合邻居特征（调用MessagePassing的mean聚合）
        Tensor aggrNeighbor = propagate(edge_index, x);

        // 2. 变换邻居和自身特征
        Tensor neighborFeat = linNeighbor.forward(aggrNeighbor);
        Tensor selfFeat = linSelf.forward(x);

        // 3. 融合特征（GraphSAGE标准：相加）
        Tensor out = neighborFeat.add(selfFeat);

        // 4. 激活 + 归一化（遵循GraphSAGE标准流程）
        out = relu(out); // 先激活再归一化
        if (this.normalize) {
            // 计算每行的L2范数（dim=-1），keepdim保证维度匹配
            Tensor norm = out.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);
            // 防止除0，最小范数限制为1e-12
            norm = norm.clamp_min(new Scalar(1e-12));
            out = out.div(norm);
        }

        return out;
    }

    // ========== 二部图前向传播（扩展方法） ==========
    /**
     * 二部图前向传播（源/目标节点特征分离）
     * @param xSrc 源节点特征 [N_src, inDim]
     * @param xDst 目标节点特征 [N_dst, inDim]
     * @param edge_index 边索引 [2, E] (src→dst)
     * @return 目标节点输出特征 [N_dst, outDim]
     */
    public Tensor forward(Tensor xSrc, Tensor xDst, Tensor edge_index) {
        // 1. 聚合源节点特征到目标节点
        long[] size = new long[]{xSrc.size(0), xDst.size(0)};
        Tensor aggrNeighbor = propagate(edge_index, xSrc, size);

        // 2. 变换并融合特征
        Tensor neighborFeat = linNeighbor.forward(aggrNeighbor);
        Tensor selfFeat = linSelf.forward(xDst);
        Tensor out = neighborFeat.add(selfFeat);

        // 3. 激活 + 归一化
        out = relu(out);
        if (this.normalize) {
            Tensor norm = out.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);
            norm = norm.clamp_min(new Scalar(1e-12));
            out = out.div(norm);
        }

        return out;
    }

    // ========== MessagePassing 基类方法实现 ==========
    /**
     * 消息函数：GraphSAGE的message就是邻居特征本身
     * 必须匹配基类签名，参数按需使用
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_j: 邻居节点特征 [E, inDim]
        return x_j;
    }

    /**
     * 聚合函数：复用基类的mean聚合（无需重写，仅为清晰展示）
     */
    @Override
    public Tensor aggregate(Tensor msg, Tensor index, long numNodes) {
        return super.aggregate(msg, index, numNodes);
    }

    // ========== 工具方法（获取线性层，方便测试） ==========
    public LinearImpl getLinNeighbor() {
        return linNeighbor;
    }

    public LinearImpl getLinSelf() {
        return linSelf;
    }
}

