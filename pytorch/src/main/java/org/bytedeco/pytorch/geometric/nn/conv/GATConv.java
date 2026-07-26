package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;

public class GATConv extends MessagePassing {
    private LinearImpl lin; // 线性变换层: [inChannels] → [heads*outChannels]
    private Tensor att;
    private long heads; // 注意力头数
    private long outChannels; // 单头输出维度
    private double negativeSlope;  // LeakyReLU负斜率
    private boolean concat; // 是否拼接多头结果（true:拼接，false:平均）
    public GATConv(long inChannels, long outChannels, long heads, double negativeSlope) {
        super("add");
        this.heads = heads;
        this.outChannels = outChannels;
        this.negativeSlope = negativeSlope;

        // 初始化线性层：输入 inChannels, 输出 heads * outChannels
        this.lin = new LinearImpl(inChannels, heads * outChannels);

        // 注意力向量 a: [1, heads, 2 * outChannels]
        this.att = torch.randn(new long[]{1, heads, 2 * outChannels});
        torch.xavier_uniform_(this.att);
        this.concat = true;

        register_module("lin", lin);
        register_parameter("att", att);
    }

    public GATConv(long inChannels, long outChannels, long heads, boolean concat, double negativeSlope) {
        super("add");
        this.heads = heads;
        this.outChannels = outChannels;
        this.negativeSlope = negativeSlope;

        // 初始化线性层：输入 inChannels, 输出 heads * outChannels
        this.lin = new LinearImpl(inChannels, heads * outChannels);

        // 注意力向量 a: [1, heads, 2 * outChannels]
        this.att = torch.randn(new long[]{1, heads, 2 * outChannels});
        torch.xavier_uniform_(this.att);
        this.concat = concat;

        register_module("lin", lin);
        register_parameter("att", att);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // 🔴 关键排查点 1: 打印输入 x 的维度
        // 如果输入是 [10, 16], 这里没问题。如果输入是 [80, 8], 说明上层传错了。
        long N = x.size(0);

        // 1. 先进行线性变换 (此时 x 是节点特征 [N, In])
        // 输出形状: [N, heads * outChannels]
        Tensor xLin = lin.forward(x);

        // 2. 重塑为多头形状: [N, heads, outChannels]
        // 🔴 关键排查点 2: 必须在进入 propagate 之前 view
        xLin = xLin.view(N, heads, outChannels);

        // 3. 开始传播
        var out = propagate(edge_index, xLin, new long[]{x.size(0), x.size(0)});
        // 4. 多头结果处理（拼接/平均）
        if (concat) {
            // 拼接：[N, heads, outChannels] → [N, heads*outChannels]
            out = out.view(N, heads * outChannels);
        } else {
            // 平均：[N, heads, outChannels] → [N, outChannels]
            out = out.mean(1);
        }

        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_i, x_j 此时已经是边维度 [E, heads, outChannels]
        // E = 80, heads = 8, outChannels = 8 -> 总维度 [80, 8, 8]

        Tensor targetIdx = edge_index.select(0, 1);

        // 计算 e_ij = a^T [Wh_i || Wh_j]
        Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), -1); // [E, heads, 2 * outChannels]

        // 与 att 做计算 [1, heads, 2 * outChannels]
        Tensor alpha = (catFeat.mul(this.att)).sum(-1); // [E, heads]
        alpha = torch.leaky_relu(alpha, new Scalar(negativeSlope));

        // 数值稳定的 Softmax
        alpha = scatter_softmax(alpha, targetIdx, numNodes);

        return x_j.mul(alpha.unsqueeze(-1));
    }

    public Tensor scatter_softmax(Tensor src, Tensor index, long numNodes) {
        Tensor maxVal = Scatter.scatter(src, index, numNodes, "max");
        Tensor out = src.sub(maxVal.index_select(0, index)).exp();
        Tensor sum = Scatter.scatter(out, index, numNodes, "add");
        return out.div(sum.index_select(0, index).add(new Scalar(1e-16)));
    }

    @Override
    public Tensor update(Tensor inputs, Tensor x) {
        // 最终输出拼接: [N, heads * outChannels]
        return inputs.view(inputs.size(0), heads * outChannels);
    }

    public LinearImpl getLin() {
        return lin;
    }



    public long getHeads() {
        return heads;
    }

    public long getOutChannels() {
        return outChannels;
    }

    public double getNegativeSlope() {
        return negativeSlope;
    }

    public boolean isConcat() {
        return concat;
    }
    public Tensor getAttParam() {
        return att;
    }
}
/**
 * 标准 Multi-head Graph Attention Network (GAT) 算子
 */
//public class GATConv extends MessagePassing {
//    private LinearImpl lin;
//    private Tensor att;
//    private long heads;
//    private long outChannels;
//    private double negativeSlope;
//
//    public GATConv(long inChannels, long outChannels, long heads, double negativeSlope) {
//        super("add");
//        this.heads = heads;
//        this.outChannels = outChannels;
//        this.negativeSlope = negativeSlope;
//
//        // 1. 权重映射层: [In] -> [Heads * Out]
//        this.lin = new LinearImpl(inChannels, heads * outChannels);
//
//        // 2. 注意力参数 a: [1, Heads, 2 * Out]
//        this.att = torch.randn(new long[]{1, heads, 2 * outChannels});
//        torch.xavier_uniform_(this.att); // 生产级初始化
//
//        register_module("lin", lin);
//        register_parameter("att", att);
//    }
//
//    public GATConv(long inChannels, long outChannels) {
//        this(inChannels, outChannels, 1, 0.2);
//    }
//
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        long N = x.size(0);
//
//        // 关键步骤: [N, C_in] -> [N, Heads, C_out]
//        // 必须显式变换维度，否则之后的拼接会报错
//        Tensor xLin = lin.forward(x).view(N, heads, outChannels);
//
//        return propagate(edge_index, xLin, null);
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // x_i, x_j shape: [Edges, Heads, outChannels]
//        Tensor targetIdx = edge_index.select(0, 1);
//
//        // 1. 计算 e_ij = a^T [Wh_i || Wh_j]
//        // 拼接特征: [Edges, Heads, 2 * outChannels]
//        Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), -1);
//
//        // 计算点积: [Edges, Heads, 2 * outChannels] * [1, Heads, 2 * outChannels] -> sum(-1) -> [Edges, Heads]
//        Tensor alpha = (catFeat.mul(this.att)).sum(-1);
//        alpha = torch.leaky_relu(alpha, new Scalar(negativeSlope));
//
//        // 2. 对每个节点的邻居进行 Softmax 归一化
//        alpha = scatter_softmax(alpha, targetIdx, numNodes);
//
//        // 3. 加权邻居特征: [Edges, Heads, outChannels] * [Edges, Heads, 1]
//        return x_j.mul(alpha.unsqueeze(-1));
//    }
//
//    /**
//     * 实现标准的数值稳定版 Scatter Softmax
//     */
//    private Tensor scatter_softmax(Tensor src, Tensor index, long numNodes) {
//        // 1. e_ij = exp(e_ij - max(e_k)) 防止数值溢出
//        Tensor maxVal = Scatter.scatter(src, index, numNodes, "max");
//        Tensor out = src.sub(maxVal.index_select(0, index)).exp();
//
//        // 2. 计算分母 sum(exp(...))
//        Tensor sum = Scatter.scatter(out, index, numNodes, "add");
//
//        // 3. 归一化，加 epsilon 防止除零
//        return out.div(sum.index_select(0, index).add(new Scalar(1e-16)));
//    }
//
//    @Override
//    public Tensor update(Tensor inputs, Tensor x) {
//        // 生产级默认: 将多头结果拼接 (Concatenate)
//        // [N, Heads, Out] -> [N, Heads * Out]
//        return inputs.view(inputs.size(0), heads * outChannels);
//    }
//}
//public class GATConv extends MessagePassing {
//
//    private LinearImpl lin;
//    private Tensor att; // 注意力向量 a
//    private long heads;
//    private long outChannels;
//    private long inChannels;
//    private double negativeSlope = 0.2;
//
//    public GATConv(long inChannels, long outChannels) {
//        this(inChannels, outChannels, 1, 0.2);
//    }
//    public GATConv(long inChannels, long outChannels, long heads, double negativeSlope) {
//        super("add");
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.heads = heads;
//        this.negativeSlope = negativeSlope;
//
//        // Linear: [N, in] -> [N, heads * out]
//        this.lin = new LinearImpl(inChannels, heads * outChannels);
//
//        // Attention Parameter: [1, heads, 2 * out]
//        this.att = torch.randn(new long[]{1, heads, 2 * outChannels});
//        xavier_uniform_(this.att);
//        register_module("lin", lin);
//        register_parameter("att", att);
//    }
//
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        long numNodes = x.size(0);
//
//        // 1. 线性变换并 Reshape 为多头
//        // [N, heads * out] -> [N, heads, out]
//        Tensor xLin = lin.forward(x).view(numNodes, heads, outChannels);
//
//        // 2. 这里的第三个参数传 edge_index 是为了在 message 里做 softmax
//        // 在我们的 MessagePassing 接口中，我们可以利用 edge_attr 槽位传递额外信息
//        return propagate(edge_index, xLin, null);
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 这里的 edge_index 是从 propagate 传下来的，不再是 null
//        Tensor targetIdx = edge_index.select(0, 1);
//        // 1. 计算 Attention Score (e_ij)
//        Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), -1);
//        Tensor alpha = (catFeat.mul(this.att)).sum(-1);
//        alpha = torch.leaky_relu(alpha, new Scalar(negativeSlope));
//
//        // 2. Softmax 归一化 (关键点：使用传进来的 targetIdx)
//        alpha = scatter_softmax(alpha, targetIdx, numNodes); // 这里的 -1 表示自动推导或传入 numNodes
//
//        return x_j.mul(alpha.unsqueeze(-1));
//    }
//
////    @Override
////    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index) {
////        // x_j, x_i shape: [E, heads, outChannels]
////
////        // 1. 拼接 source 和 target 的特征
////        // cat([x_i, x_j]) -> [E, heads, 2 * out]
////        Tensor catFeat = torch.cat(new TensorArrayRef(new TensorVector(new Tensor[]{x_i, x_j})), -1);
////
////        // 2. 计算 Attention Score
////        // (cat * att).sum(-1) -> [E, heads]
////        // 注意：这里需要处理 att 的维度广播
////        Tensor alpha = (catFeat.mul(this.att)).sum(-1);
////
////        // 3. LeakyReLU
////        alpha = torch.leaky_relu(alpha, new Scalar(0.2));
////
////        // 4. Softmax (这就比较难了，需要对每个目标节点 i 的邻居做 softmax)
////        // PyG 使用 softmax(src, index)。我们需要手动实现基于 index 的 softmax
////        // 这里简化：直接使用 sigmoid 模拟，或者需要实现 scatter_softmax
//////        alpha = torch.sigmoid(alpha); // 这是一个简化替代方案！
////        alpha = scatter_softmax(alpha, edge_index.select(0, 1), x_j.size(0));
////
////        // 5. 加权
////        // x_j [E, heads, out] * alpha [E, heads, 1]
////        return x_j.mul(alpha.unsqueeze(-1));
////    }
//
//    /**
//     * 实现标准的 Scatter Softmax
//     */
//    private Tensor scatter_softmax(Tensor src, Tensor index, long numNodes) {
//        // 1. 减去最大值以保证数值稳定性 (Optional but recommended)
//        Tensor maxVal = Scatter.scatter(src, index, numNodes, "max");
//        Tensor temp = src.sub(maxVal.index_select(0, index));
//
//        // 2. Exp
//        Tensor exp = temp.exp();
//
//        // 3. Sum
//        Tensor sumExp = Scatter.scatter(exp, index, numNodes, "add");
//
//        // 4. Divide: exp / sumExp[index]
//        return exp.div(sumExp.index_select(0, index).add(new Scalar(1e-16))); // 加个 epsilon 防 0
//    }
//
////    public Tensor aggregate(Tensor inputs, Tensor index, long dimSize) {
////        return Scatter.scatter(inputs, index, dimSize, this.aggr);
////    }
//    
//    @Override
//    public Tensor update(Tensor inputs, Tensor x) {
//        // inputs shape: [N, heads, out]
//        // 如果是 concat 模式：
//        long numNodes = inputs.size(0);
//        return inputs.view(numNodes, heads * outChannels);
//    }
//
//}