package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;

/**
 * 修复维度匹配问题的稠密图注意力卷积层（继承 MessagePassing）
 * 输入：x [B, N, in_channels], adj [B, N, N]
 * 输出：out [B, N, heads * out_channels]
 */
public class DenseGATConv extends MessagePassing {
    public LinearImpl lin;
    private Parameter attSrc, attDst; // 可训练的注意力参数
    private long heads;
    private long outChannels;
    public Tensor alpha; // 保存注意力权重（供 message/aggregate 使用）
    private Tensor xFeat; // 保存投影后的特征

    public DenseGATConv(long inChannels, long outChannels, long heads) {
        super(); // 调用 MessagePassing 父类构造器
        this.heads = heads;
        this.outChannels = outChannels;

        // 1. 线性投影层：inChannels -> heads * outChannels
        this.lin = new LinearImpl(inChannels, heads * outChannels);
        register_module("lin", lin); // 注册子模块

        // 2. 注意力参数：[1, heads, outChannels]，包装为 Parameter
        Tensor attSrcTensor = torch.randn(new long[]{1, heads, outChannels});
        this.attSrc = new Parameter(attSrcTensor);
        register_parameter("attSrc", this.attSrc);

        Tensor attDstTensor = torch.randn(new long[]{1, heads, outChannels});
        this.attDst = new Parameter(attDstTensor);
        register_parameter("attDst", this.attDst);
    }

    /**
     * 前向传播（核心：调用自定义 propagate 完成消息传递）
     * @param x 节点特征 [B, N, in_channels]
     * @param adj 邻接矩阵 [B, N, N] (0/1 矩阵，表示节点连接关系)
     * @return 输出特征 [B, N, heads * out_channels]
     */
    public Tensor forward(Tensor x, Tensor adj) {
        // 输入维度校验
        if (x.dim() != 3) {
            throw new IllegalArgumentException("x 必须是 3 维张量 [B, N, in_channels]，当前维度：" + x.dim());
        }
        if (adj.dim() != 3) {
            throw new IllegalArgumentException("adj 必须是 3 维张量 [B, N, N]，当前维度：" + adj.dim());
        }

        long B = x.size(0);
        long N = x.size(1);

        // 1. 线性投影：[B, N, in] -> [B, N, heads*out] -> [B, N, heads, out]
        this.xFeat = lin.forward(x).view(B, N, heads, outChannels);

        // 2. 计算注意力分数：(x * att).sum(-1) -> [B, N, heads]
        // 显式传入维度数组，适配 JavaCPP 签名
        Tensor alphaSrc = xFeat.mul(attSrc).sum(new long[]{3}, false, new ScalarTypeOptional());
        Tensor alphaDst = xFeat.mul(attDst).sum(new long[]{3}, false, new ScalarTypeOptional());

        // 3. 计算注意力 logits：alpha_src[i] + alpha_dst[j] -> [B, N, N, heads]
        Tensor logits = alphaSrc.unsqueeze(2).add(alphaDst.unsqueeze(1));
        logits = torch.leaky_relu(logits, new Scalar(0.2));

        // 4. 邻接矩阵掩码：adj=0 的位置设为负无穷（避免 softmax 后权重不为 0）
        Tensor mask = adj.unsqueeze(3).eq(new Scalar(0));
        logits.masked_fill_(mask, new Scalar(-1e9));

        // 5. Softmax 归一化得到注意力权重（dim=2 是邻居维度）
        this.alpha = torch.softmax(logits, 2);

        // 6. 调用自定义 propagate 方法完成消息传递
        Tensor out = propagate(adj, xFeat);

        // 7. 拼接 heads：[B, N, heads, out] -> [B, N, heads*out]
        // 先校验维度，避免 reshape 错误
        long[] outSizes = out.sizes().vec().get();
        if (outSizes.length != 4 || outSizes[0] != B || outSizes[1] != N || outSizes[2] != heads || outSizes[3] != outChannels) {
            throw new RuntimeException("聚合后维度异常：预期 [B,N,heads,out] = [" + B + "," + N + "," + heads + "," + outChannels + "]，实际：" + out);
        }
        out = out.reshape(B, N, heads * outChannels);
        return out;
    }

    /**
     * 自定义 message 方法（适配稠密图，构建注意力加权的邻居消息）
     * @param xFeat 投影后的特征 [B, N, heads, out]
     * @return 消息张量 [B, N, N, heads, out]
     */
    private Tensor denseMessage(Tensor xFeat) {
        long B = xFeat.size(0);
        long N = xFeat.size(1);

        // 扩展维度：[B, N, heads, out] -> [B, N, 1, heads, out]（适配邻居维度）
        Tensor xFeatExpanded = xFeat.unsqueeze(2);
        // 注意力权重扩展：[B, N, N, heads] -> [B, N, N, heads, 1]（适配特征维度）
        Tensor alphaExpanded = alpha.unsqueeze(4);

        // 消息 = 注意力权重 * 邻居特征 -> [B, N, N, heads, out]
        Tensor msg = xFeatExpanded.mul(alphaExpanded);
        return msg;
    }

    /**
     * 自定义 aggregate 方法（对邻居维度求和）
     * @param message 消息张量 [B, N, N, heads, out]
     * @return 聚合后的特征 [B, N, heads, out]
     */
    private Tensor denseAggregate(Tensor message) {
        // 对邻居维度（dim=2）求和，保持维度（false）
        return message.sum(new long[]{2}, false, new ScalarTypeOptional());
    }

    /**
     * 重载 propagate 适配稠密图（核心：处理矩阵形式的邻接关系）
     * @param adj 邻接矩阵 [B, N, N]
     * @param xFeat 投影后的特征 [B, N, heads, out]
     * @return 聚合后的特征 [B, N, heads, out]
     */
    public Tensor propagate(Tensor adj, Tensor xFeat) {
        // 1. 构建消息：调用自定义 denseMessage 方法
        Tensor msg = denseMessage(xFeat);
        // 2. 聚合消息：调用自定义 denseAggregate 方法
        Tensor agg = denseAggregate(msg);
        return agg;
    }

    // 基类要求的 message 方法（稀疏图用，稠密图不使用，返回空即可）
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return null;
    }

    // 释放资源（JavaCPP 内存管理）
    @Override
    public void close() {
        if (lin != null) lin.close();
        if (attSrc != null) attSrc.close();
        if (attDst != null) attDst.close();
        if (alpha != null) alpha.close();
        if (xFeat != null) xFeat.close();
        super.close();
    }
}
/**
 * 继承 MessagePassing 的稠密图注意力卷积层（DenseGATConv）
 * 输入：x [B, N, in_channels], adj [B, N, N]
 * 输出：out [B, N, heads * out_channels]
 */
//public class DenseGATConv extends MessagePassing {
//    public LinearImpl lin;
//    private Parameter attSrc, attDst; // 可训练的注意力参数
//    private long heads;
//    private long outChannels;
//    public Tensor alpha; // 保存注意力权重（供 message/aggregate 使用）
//    private Tensor xFeat; // 保存投影后的特征
//
//    public DenseGATConv(long inChannels, long outChannels, long heads) {
//        super(); // 调用 MessagePassing 父类构造器
//        this.heads = heads;
//        this.outChannels = outChannels;
//
//        // 1. 线性投影层：inChannels -> heads * outChannels
//        this.lin = new LinearImpl(inChannels, heads * outChannels);
//        register_module("lin", lin); // 注册子模块
//
//        // 2. 注意力参数：[1, heads, outChannels]，包装为 Parameter
//        Tensor attSrcTensor = torch.randn(new long[]{1, heads, outChannels});
//        this.attSrc = new Parameter(attSrcTensor);
//        register_parameter("attSrc", this.attSrc);
//
//        Tensor attDstTensor = torch.randn(new long[]{1, heads, outChannels});
//        this.attDst = new Parameter(attDstTensor);
//        register_parameter("attDst", this.attDst);
//    }
//
//    /**
//     * 前向传播（核心：调用 propagate 完成消息传递）
//     * @param x 节点特征 [B, N, in_channels]
//     * @param adj 邻接矩阵 [B, N, N] (0/1 矩阵，表示节点连接关系)
//     * @return 输出特征 [B, N, heads * out_channels]
//     */
//    public Tensor forward(Tensor x, Tensor adj) {
//        // 输入维度校验
//        if (x.dim() != 3) {
//            throw new IllegalArgumentException("x 必须是 3 维张量 [B, N, in_channels]，当前维度：" + x.dim());
//        }
//        if (adj.dim() != 3) {
//            throw new IllegalArgumentException("adj 必须是 3 维张量 [B, N, N]，当前维度：" + adj.dim());
//        }
//
//        long B = x.size(0);
//        long N = x.size(1);
//
//        // 1. 线性投影：[B, N, in] -> [B, N, heads*out] -> [B, N, heads, out]
//        this.xFeat = lin.forward(x).view(B, N, heads, outChannels);
//
//        // 2. 计算注意力分数：(x * att).sum(-1) -> [B, N, heads]
//        Tensor alphaSrc = xFeat.mul(attSrc).sum(new long[]{3}, false, new ScalarTypeOptional());
//        Tensor alphaDst = xFeat.mul(attDst).sum(new long[]{3}, false, new ScalarTypeOptional());
//
//        // 3. 计算注意力 logits：alpha_src[i] + alpha_dst[j] -> [B, N, N, heads]
//        Tensor logits = alphaSrc.unsqueeze(2).add(alphaDst.unsqueeze(1));
//        logits = torch.leaky_relu(logits, new Scalar(0.2));
//
//        // 4. 邻接矩阵掩码：adj=0 的位置设为负无穷
//        Tensor mask = adj.unsqueeze(3).eq(new Scalar(0));
//        logits.masked_fill_(mask, new Scalar(-1e9));
//
//        // 5. Softmax 归一化得到注意力权重
//        this.alpha = torch.softmax(logits, 2);
//
//        // 6. 调用 MessagePassing 的 propagate 方法完成消息传递
//        // 稠密图中：propagate 接收邻接矩阵、投影特征等，触发 message -> aggregate
//        Tensor out = propagate(adj, xFeat);
//
//        // 7. 拼接 heads：[B, N, heads, out] -> [B, N, heads*out]
//        out = out.reshape(B, N, heads * outChannels);
//        return out;
//    }
//
//    /**
//     * MessagePassing 必须实现：构建消息（注意力加权的邻居特征）
//     * 稠密场景下：x_j 是所有邻居的特征矩阵，乘以注意力权重 alpha
//     * @param args 传入的参数（xFeat: [B, N, heads, out]）
//     * @return 消息张量 [B, N, N, heads, out]
//     */
////    @Override
//    protected Tensor message(Tensor... args) {
//        Tensor xFeat = args[0]; // [B, N, heads, out]
//        long B = xFeat.size(0);
//        long N = xFeat.size(1);
//
//        // xFeat: [B, N, heads, out] -> [B, N, 1, heads, out]
//        Tensor xFeatExpanded = xFeat.unsqueeze(2);
//        // alpha: [B, N, N, heads] -> [B, N, N, heads, 1]
//        Tensor alphaExpanded = alpha.unsqueeze(4);
//
//        // 消息 = 注意力权重 * 邻居特征 -> [B, N, N, heads, out]
//        return xFeatExpanded.mul(alphaExpanded);
//    }
//
//    /**
//     * MessagePassing 必须实现：聚合消息（对邻居维度求和）
//     * @param message 消息张量 [B, N, N, heads, out]
//     * @return 聚合后的特征 [B, N, heads, out]
//     */
////    @Override
//    protected Tensor aggregate(Tensor message) {
//        // 对邻居维度（dim=2）求和：[B, N, N, heads, out] -> [B, N, heads, out]
//        return message.sum(new long[]{2}, false,new ScalarTypeOptional());
//    }
//
//    /**
//     * 重载 propagate 适配稠密图（核心：处理矩阵形式的邻接关系）
//     * @param adj 邻接矩阵 [B, N, N]
//     * @param xFeat 投影后的特征 [B, N, heads, out]
//     * @return 聚合后的特征 [B, N, heads, out]
//     */
//    public Tensor propagate(Tensor adj, Tensor xFeat) {
//        // 1. 构建消息：调用 message 方法
//        Tensor msg = message(xFeat);
//        // 2. 聚合消息：调用 aggregate 方法
//        Tensor agg = aggregate(msg);
//        return agg;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        return null;
//    }
//
//    // 释放资源（JavaCPP 内存管理）
//    @Override
//    public void close() {
//        if (lin != null) lin.close();
//        if (attSrc != null) attSrc.close();
//        if (attDst != null) attDst.close();
//        if (alpha != null) alpha.close();
//        if (xFeat != null) xFeat.close();
//        super.close();
//    }
//}

//package org.bytedeco.pytorch.geometric.nn.conv;
//
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.nn.Module;
//import org.bytedeco.pytorch.global.torch;
//
//public class DenseGATConv extends MessagePassing {
//    private LinearImpl lin;
//    private Tensor attSrc, attDst; // a_src, a_dst Parameter
//    private long heads;
//    private long outChannels;
//
//    public DenseGATConv(long inChannels, long outChannels, long heads) {
//        this.heads = heads;
//        this.outChannels = outChannels;
//
//        // W: In -> Heads * Out
//        this.lin = new LinearImpl(inChannels, heads * outChannels);
//
//        // Attention Vector: [1, Heads, Out]
//        this.attSrc = torch.randn(new long[]{1, heads, outChannels}); // new Parameter(
//        this.attDst = torch.randn(new long[]{1, heads, outChannels}); // new Parameter(
//
//        register_module("lin", lin);
//        register_parameter("attSrc", attSrc);
//        register_parameter("attDst", attDst);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor adj) {
//        long B = x.size(0);
//        long N = x.size(1);
//
//        // 1. Projection: [B, N, H * C] -> [B, N, H, C]
//        Tensor xFeat = lin.forward(x).view(B, N, heads, outChannels);
//
//        // 2. Compute Attention Logits
//        // alpha_src = (x * attSrc).sum(-1) -> [B, N, H]
//        Tensor alphaSrc = xFeat.mul(attSrc).sum(new long[]{3}, false, new ScalarTypeOptional());
//        Tensor alphaDst = xFeat.mul(attDst).sum(new long[]{3}, false, new ScalarTypeOptional());
//
//        // Logits[i, j] = alpha_src[i] + alpha_dst[j]
//        // [B, N, 1, H] + [B, 1, N, H] -> [B, N, N, H]
//        Tensor logits = alphaSrc.unsqueeze(2).add(alphaDst.unsqueeze(1));
//        logits = torch.leaky_relu(logits, new Scalar(0.2));
//
//        // 3. Mask with Adjacency (Force A_ij=0 to -inf)
//        // Adj: [B, N, N] -> [B, N, N, 1]
//        Tensor mask = adj.unsqueeze(3).eq(new Scalar(0));
//        logits.masked_fill_(mask, new Scalar(Float.NEGATIVE_INFINITY));
//
//        // 4. Softmax -> Attention Weights
//        Tensor alpha = torch.softmax(logits, 2); // dim 2 is neighbors (j)
//
//        // 5. Aggregate
//        // Out = sum_j ( alpha_ij * x_j )
//        // Alpha: [B, N, N, H]
//        // X: [B, N, H, C] -> [B, 1, N, H, C]
//        // Alpha -> [B, N, N, H, 1]
//        // Product -> [B, N, N, H, C] -> sum(dim=2) -> [B, N, H, C]
//
//        // Einsum is easier: 'bnhj, bnjhc -> bnhc'? No JavaCPP einsum easy to use.
//        // Use matmul per head: 
//        // Permute Alpha: [B, H, N, N]
//        // Permute X: [B, H, N, C]
//        // Matmul: [B, H, N, N] @ [B, H, N, C] -> [B, H, N, C]
//
//        // 5. Aggregate (修复 Permute 逻辑)
//        // alpha: [B, N, N, H] -> [B, H, N, N]
//        Tensor alphaPerm = alpha.permute(0, 3, 1, 2);
//
//        // xFeat: [B, N, H, C] -> [B, H, N, C]  <-- 关键改动在这里
//        // 原来你用了 (0, 3, 1, 2) 导致 C 跑到了 H 的位置
//        Tensor xPerm = xFeat.permute(0, 2, 1, 3);
//
//        // [B, H, N, N] @ [B, H, N, C] -> [B, H, N, C]
//        Tensor out = alphaPerm.matmul(xPerm);
////        Tensor alphaPerm = alpha.permute(0, 3, 1, 2);
////        Tensor xPerm = xFeat.permute(0, 3, 1, 2);
////        Tensor out = alphaPerm.matmul(xPerm);
//
//        // 6. Concat Heads: [B, N, H * C]
//        return out.permute(0, 2, 1, 3).reshape(B, N, heads * outChannels);
//    }
//
//    /**
//     * 必须匹配基类签名：(x_j, x_i, edge_index, edge_attr)
//     * 哪怕 SAGE 只需要 x_j，参数也必须写全！
//     */
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // GraphSAGE 的 message 就是邻居特征本身
//        // 如果以后要支持带权重的 SAGE，可以在这里处理 edge_attr
//        return x_j;
//    }
//}