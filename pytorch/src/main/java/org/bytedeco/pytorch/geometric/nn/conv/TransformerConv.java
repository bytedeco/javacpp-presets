package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.utils.Scatter;
//import org.gnn.framework.nn.org.bytedeco.pytorch.geometric.nn.conv.MessagePassing;


public class TransformerConv extends MessagePassing {
    private LinearImpl linKey, linQuery, linValue;
    private LinearImpl linSkip;
    private long heads;
    private long outChannels;

    public TransformerConv(long inChannels, long outChannels, long heads) {
        super("add"); // Base aggregation, but we control alpha manually
        this.heads = heads;
        this.outChannels = outChannels;

        linKey = new LinearImpl(inChannels, heads * outChannels);
        linQuery = new LinearImpl(inChannels, heads * outChannels);
        linValue = new LinearImpl(inChannels, heads * outChannels);
        linSkip = new LinearImpl(inChannels, heads * outChannels); // Skip connection

        register_module("lin_query", linQuery);
        register_module("lin_key", linKey);
        register_module("lin_value", linValue);
        register_module("lin_skip", linSkip);
//        register_module("linKey", linKey);
//        register_module("linQuery", linQuery);
//        register_module("linValue", linValue);
//        register_module("linSkip", linSkip);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        long N = x.size(0);

        // 1. 预计算所有节点的 Q, K, V (效率提升关键)
        // [N, In] -> [N, H, C]
        Tensor q = linQuery.forward(x).view(N, heads, outChannels);
        Tensor k = linKey.forward(x).view(N, heads, outChannels);
        Tensor v = linValue.forward(x).view(N, heads, outChannels);

        // 2. 传播
        // 我们需要把 q, k, v 都传下去。在 Java 契约中，
        // 我们可以把 q 放在 x 的位置，把 (k, v) 封装或分次处理。
        // 这里推荐：propagate 传入 v，将 q 和 k 作为额外的 edge_attr 处理（或通过其他方式）
        // 这里的演示为了符合你之前的 propagate(edge_index, x, edge_attr) 契约：
        // 我们假设 x 传的是 v，edge_attr 传的是计算好的 alpha 或相关张量。

        // 更标准的做法是重载 propagate 支持多参数，或者像下面这样：
        return propagate_transformer(edge_index, q, k, v, x);
    }
    
    public Tensor forward2(Tensor x, Tensor edge_index) {
        long numNodes = x.size(0);

        // Calculate Q, K, V
        Tensor k = linKey.forward(x).view(numNodes, heads, outChannels);
        Tensor q = linQuery.forward(x).view(numNodes, heads, outChannels);
        Tensor v = linValue.forward(x).view(numNodes, heads, outChannels);

        // Propagate needs specific Q and K/V
        // 这里为了简化，我们把 (k, v) 打包传给 message，或者在 message 里重新通过 index 取
        // 实际上 org.bytedeco.pytorch.geometric.nn.conv.MessagePassing 很难传多个 Tensor，通常的做法是只传 V，
        // K 和 Q 在 message 内部通过 x_j 和 x_i 重新计算 (稍微慢点但结构清晰)

        // 重新策略：我们在 message 里直接用 x_i 和 x_j 计算
        Tensor out = propagate(edge_index, x); // 此处 x 还是原始特征

        // Residual connection
        Tensor skip = linSkip.forward(x);//.view(numNodes, heads, outChannels);
        return out.add(skip);
    }

//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index) {
//        // x_j: neighbor features [E, In]
//        // x_i: target features [E, In]
//
//        long numEdges = x_j.size(0);
//
//        // 1. Recalculate K_j, Q_i inside message (simplest implementation)
//        Tensor K_j = linKey.forward(x_j).view(numEdges, heads, outChannels);
//        Tensor Q_i = linQuery.forward(x_i).view(numEdges, heads, outChannels);
//        Tensor V_j = linValue.forward(x_j).view(numEdges, heads, outChannels);
//
//        // 2. Attention Score = (Q_i * K_j) / sqrt(d)
//        // sum(-1) 做点积
//        Tensor alpha = Q_i.mul(K_j).sum(new long[]{ -1 }, false,new ScalarTypeOptional());
//
//        // Scale by sqrt(out_channels)
//        // 重点：new Scalar
//        alpha = alpha.div(new Scalar(Math.sqrt(outChannels)));
//
//        // 3. Softmax
//        // 注意：这里需要 softmax over neighbors。
//        // 由于我们没有实现 scatter_softmax，这里暂时用 sigmoid 模拟 (简化版)
//        // 或者是标准的 softmax_v2 如果维度允许
//        alpha = sigmoid(alpha); // Placeholder for softmax(alpha, index)
//
//        // 4. Weight V_j
//        // alpha [E, Heads] -> [E, Heads, 1]
//        return V_j.mul(alpha.unsqueeze(-1));
//    }

    // 自定义内部传播逻辑，以处理 Transformer 的多张量需求
    private Tensor propagate_transformer(Tensor edge_index, Tensor q, Tensor k, Tensor v, Tensor x_original) {
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);
        long numNodes = q.size(0); // 确定节点总数 N
        // Lift
        Tensor q_i = q.index_select(0, targetIdx); // [E, H, C]
        Tensor k_j = k.index_select(0, sourceIdx); // [E, H, C]
        Tensor v_j = v.index_select(0, sourceIdx); // [E, H, C]

        // 调用自定义 message
        Tensor msg = message_transformer(q_i, k_j, v_j, edge_index,numNodes);

        Tensor out = aggregate(msg, targetIdx, q.size(0));

        // Skip connection + Reshape
        Tensor skip = linSkip.forward(x_original).view(q.size(0), heads, outChannels);
        return out.add(skip).view(q.size(0), heads * outChannels);
    }

    private Tensor message_transformer(Tensor q_i, Tensor k_j, Tensor v_j, Tensor edge_index, long numNodes) {
        // 1. Attention Score: (q_i * k_j).sum(-1)
        Tensor alpha = q_i.mul(k_j).sum(-1).div(new Scalar(Math.sqrt(outChannels)));

        // 2. Softmax over target nodes (edge_index[1])
        alpha = scatter_softmax(alpha, edge_index.select(0, 1), numNodes);//-1);

        // 3. Weight Values: [E, H, C] * [E, H, 1]
        return v_j.mul(alpha.unsqueeze(-1));
    }
    @Override
    public Tensor update(Tensor inputs, Tensor x) {
        // inputs: [N, heads, out] -> [N, heads * out]
        return inputs.view(x.size(0), heads * outChannels);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 实现基类空方法以防止 NPE，逻辑已移至 message_transformer
        return x_j;
    }
    private Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
        // 这里的实现与 GAT 中的一致
        Tensor maxVal = Scatter.scatter(src, index, dimSize, "max");
        Tensor out = src.sub(maxVal.index_select(0, index)).exp();
        Tensor sum = Scatter.scatter(out, index, dimSize, "add");
        return out.div(sum.index_select(0, index).add(new Scalar(1e-16)));
    }
    
}
