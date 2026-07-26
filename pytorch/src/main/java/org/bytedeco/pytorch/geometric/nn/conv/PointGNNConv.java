package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

/**
 * 严格使用 LinearImpl 规范实现 torch_geometric.nn.conv.PointGNNConv
 * 特点：引入对齐偏移 delta_pos 以增强 3D 几何特征的提取。
 */



/**
 * 完整原装实现的 PointGNNConv
 */
public class PointGNNConv extends MessagePassing {
    private SequentialImpl mlpH;
    private SequentialImpl mlpF;
    private SequentialImpl mlpG;

    public PointGNNConv(SequentialImpl mlpH, SequentialImpl mlpF, SequentialImpl mlpG) {
        super("max"); // 论文建议使用 max 聚合
        this.mlpH = mlpH;
        this.mlpF = mlpF;
        this.mlpG = mlpG;

        // 必须注册，否则子模块参数不会被优化器追踪，且 JNI 对象生命周期受限
        register_module("mlp_h", mlpH);
        register_module("mlp_f", mlpF);
        register_module("mlp_g", mlpG);
    }

    public Tensor forward(Tensor x, Tensor pos, Tensor edge_index) {
        // 1. 调用 mlpH 预测对齐偏移 [N, 3]
        // 注意：原装调用方式使用 asSequential()
        Tensor deltaPos = mlpH.asSequential().forward(x);

        // 2. 启动消息传递
        // 我们需要传递 x, pos 和 deltaPos，它们会被自动解包为 x_j, pos_j, pos_i, deltaPos_i
        return propagate(edge_index, x, pos, deltaPos);
    }

    public Tensor propagate(Tensor edge_index, Tensor x, Tensor pos, Tensor deltaPos) {
        // 1. 获取源节点(j)和目标节点(i)的索引
        Tensor row = edge_index.select(0, 0); // Source
        Tensor col = edge_index.select(0, 1); // Destination
        long numNodes = x.size(0);

        // 2. 准备邻居特征和坐标
        Tensor x_j = x.index_select(0, row);
        Tensor pos_j = pos.index_select(0, row);
        Tensor pos_i = pos.index_select(0, col);
        Tensor deltaPos_i = deltaPos.index_select(0, col);

        // 3. 调用 message 函数进行特征融合
        Tensor msg = message(x_j, pos_j, pos_i, deltaPos_i, numNodes);

        // 4. 执行聚合 (Max)
        Tensor aggrOut = aggregate(msg, col, numNodes);

        // 5. 调用 update 进行残差更新
        return update(aggrOut, x);
    }
    /**
     * 覆盖基类 message，防止默认的 mul 行为导致维度冲突 (16 vs 3)
     */
    @Override
    public Tensor message(Tensor x_j, Tensor pos_j, Tensor pos_i, Tensor deltaPos_i, long numNodes) {
        // 计算平移不变性的相对坐标: rel_pos = pos_j - (pos_i + deltaPos_i)
        Tensor relPos = pos_j.sub(pos_i.add(deltaPos_i));

        // 核心修复：将 16 维特征与 3 维坐标拼接，得到 19 维输入
        TensorVector v = new TensorVector(x_j, relPos);
        Tensor fInput = torch.cat(v, -1);

        // 调用 mlpF (原装方式)
        return mlpF.asSequential().forward(fInput);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }

    @Override
    public Tensor update(Tensor aggrOut, Tensor x) {
        // 最终更新：中心节点特征 x + mlpG(聚合后的消息)
        Tensor updated = mlpG.asSequential().forward(aggrOut);
  

        // 核心修复：检查维度是否匹配以进行残差相加
        if (x.size(1) == updated.size(1)) {
            return x.add(updated);
        } else {
            // 如果维度不匹配 (16 vs 32)，直接返回变换后的特征
            // 或者你可以选择实现一个 linResidual 来对齐 x 的维度
            return updated;
        }
    }
}
//public class PointGNNConv extends MessagePassing {
//    private SequentialImpl mlpH; // 预测偏移量 delta_pos_i: x_i -> R^3
//    private SequentialImpl mlpF; // 消息函数: (x_j, rel_pos) -> hidden
//    private SequentialImpl mlpG; // 聚合更新函数: aggregated -> out_channels
//
//    public PointGNNConv(SequentialImpl mlpH, SequentialImpl mlpF, SequentialImpl mlpG) {
//        super("max"); // 论文通常建议使用 max 聚合
//        this.mlpH = mlpH;
//        this.mlpF = mlpF;
//        this.mlpG = mlpG;
//
//        // 必须注册子模块，确保内部的 LinearImpl 被追踪
//        register_module("mlp_h", mlpH);
//        register_module("mlp_f", mlpF);
//        register_module("mlp_g", mlpG);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, (Tensor)null);
//    }
//    
//    /**
//     * @param x          节点特征 [N, in_channels]
//     * @param pos        节点 3D 坐标 [N, 3]
//     * @param edge_index 边索引 [2, E] (通常基于半径搜索构建)
//     */
//    public Tensor forward(Tensor x, Tensor pos, Tensor edge_index) {
//        // 1. 计算对齐偏移 delta_pos
//        // mlpH 必须包含 LinearImpl，将特征映射到 3 维空间
//        Tensor deltaPos = mlpH.asSequential().forward(x); // [N, 3] asSequential()
//
//        // 2. 执行消息传递
//        // 注意：我们需要将原始坐标和偏移量都传入消息函数
//        return propagate(edge_index, x, pos, deltaPos);
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor pos_j, Tensor pos_i, Tensor deltaPos_i, long numNodes) {
//        // 计算相对对齐坐标: (pos_j - (pos_i + deltaPos_i))
//        // 这种平移不变性对于识别不同位置的相同设备至关重要
//        Tensor relPos = pos_j.sub(pos_i.add(deltaPos_i));
//
//        // 拼接邻居特征与相对坐标作为 mlpF 的输入
//        Tensor fInput = torch.cat(new TensorVector(x_j, relPos), -1);
//
//        // mlpF 内部应使用 LinearImpl 投影
//        return mlpF.forward(fInput); //.asSequential()
//    }
//
//    @Override
//    public Tensor update(Tensor aggrOut, Tensor x) {
//        // 聚合后的特征通过 mlpG 映射回目标空间
//        // 最终输出通常采用残差结构：x + mlpG(aggrOut)
//        return x.add(mlpG.forward(aggrOut)); //.asSequential().
//    }
//}