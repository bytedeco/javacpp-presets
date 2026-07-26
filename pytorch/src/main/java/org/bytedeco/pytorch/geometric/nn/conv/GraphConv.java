package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Tensor;

public class GraphConv extends MessagePassing {

    private LinearImpl linRoot; // 权重 W1: 处理自身 x
    private LinearImpl linRel;  // 权重 W2: 处理邻居聚合 sum(x_j)


    /**
     * @param inDimSrc 源节点特征维度 (如 Author: 32)
     * @param inDimDst 目标节点特征维度 (如 Paper: 16)
     * @param outDim   输出特征维度 (如 64)
     */
    public GraphConv(long inDimSrc, long inDimDst, long outDim) {
        super();
        // linRel 必须匹配源节点的维度 (inDimSrc)
        this.linRel = register_module("linRel", new LinearImpl(inDimSrc, outDim));
        // linRoot 必须匹配目标节点的维度 (inDimDst)
        this.linRoot = register_module("linRoot", new LinearImpl(inDimDst, outDim));
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }

    // GraphConv.java 内部
    public Tensor forward(Tensor xSrc, Tensor xDst, Tensor edgeIndex) {
        // 1. 处理源节点 (Neighbors)
        Tensor rel = linRel.forward(xSrc); // 对应权重 linRel

        // 2. 传播：将 50 个 author 的特征聚合到 100 个 paper 上
        long[] size = new long[]{xSrc.size(0), xDst.size(0)};
        Tensor out = propagate(edgeIndex, rel, size);

        // 3. 处理目标节点自身 (Root)
        Tensor root = linRoot.forward(xDst); // 对应权重 linRoot

        return out.add(root);
    }

   

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 在 GraphConv 中，edge_attr 承载的是 edgeWeight
        if (edge_attr != null) {
            // [E, C] * [E, 1]
            return x_j.mul(edge_attr.view(-1, 1));
        }
        return x_j;
    }

    // update 方法使用基类默认的 (直接返回 inputs) 即可
}


//import org.gnn.framework.nn.org.bytedeco.pytorch.geometric.nn.conv.MessagePassing;
//    private long outChannels;

//    public GraphConv(long inChannels, long outChannels) {
//        // org.bytedeco.pytorch.geometric.nn.conv.GraphConv 通常使用 "add" (sum) 聚合，也可以是 "mean"
//        super("add");
//        this.outChannels = outChannels;
//
//        // 初始化两个独立的线性层
//        this.linRoot = new LinearImpl(inChannels, outChannels);
//        this.linRel = new LinearImpl(inChannels, outChannels);
//
//        // 注册模块以便 PyTorch 追踪参数
//        register_module("linRoot", linRoot);
//        register_module("linRel", linRel);
//    }
/**
 * 实现标准的 org.bytedeco.pytorch.geometric.nn.conv.GraphConv
 * 公式: Out = Lin_Root(x) + Lin_Rel( Sum(Neighbors) )
 */

//public Tensor forward2(Tensor x, Tensor edge_index, Tensor edgeWeight) {
//    // 1. 计算邻居聚合 (Aggregate Neighbors)
//    // propagate 流程:
//    // -> message(x_j) 返回 x_j
//    // -> aggregate(msg) 对 x_j 求和 -> 得到 [N, inChannels]
//    // -> update(aggr) 直接返回 -> 得到 aggrOut
////        Tensor aggrOut = propagate(edge_index, x);
////
////        // 2. 对邻居信息应用变换 W2
////        // [N, in] -> [N, out]
////        Tensor neighborFeat = linRel.forward(aggrOut);
////
////        // 3. 对自身信息应用变换 W1
////        // [N, in] -> [N, out]
////        Tensor selfFeat = linRoot.forward(x);
////
////        // 4. 融合两者 (Residual like connection)
////        return neighborFeat.add(selfFeat);
//    // 1. 先进行线性变换 (Standard PyG style: x * W)
//    // 这样做的好处是变换只发生 N 次，而不是 E 次
//    Tensor xRel = linRel.forward(x);
//    Tensor xRoot = linRoot.forward(x);
//
//    // 2. 传播邻居信息
//    // 将 edgeWeight 传入，让 message 处理
//    Tensor out = propagate(edge_index, xRel, edgeWeight);
//
//    // 3. 加上自身特征变换 (W1 * x_i)
//    out = out.add(xRoot);
//
//    return out;
//}


//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edgeWeight) {
//        // org.bytedeco.pytorch.geometric.nn.conv.GraphConv 的消息极其简单，就是源节点特征
//        // 如果这里要支持边权重 (Edge Weight)，就在这里做乘法: x_j * edge_weight
//        // 如果提供了边权重，进行加权
//        if (edgeWeight != null) {
//            // x_j: [E, outChannels], edgeWeight: [E]
//            // 需要 view(-1, 1) 触发广播
//            return x_j.mul(edgeWeight.view(-1, 1));
//        }
//        return x_j;
//    }
/**
 * 必须匹配基类签名：(x_j, x_i, edge_index, edge_attr)
 */