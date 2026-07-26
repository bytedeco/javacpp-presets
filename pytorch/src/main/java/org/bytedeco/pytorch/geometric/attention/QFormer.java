package org.bytedeco.pytorch.geometric.attention;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.c10.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.MultiheadAttentionImpl;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.nn.norm.LayerNorm;

import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.relu;

public class QFormer extends org.bytedeco.pytorch.nn.Module {
    private Tensor queryTokens;
    private MultiheadAttentionImpl crossAttn; // 注意：使用 MultiheadAttention 而非 Impl 包装类
    private LinearImpl ffn1, ffn2;
    private LayerNorm ln1, ln2;
    private long dim;

    public QFormer(long dim, long numHeads, long numQueries) {
        super();
        this.dim = dim;

        // 1. Learnable Queries: [numQueries, dim]
        // 使用 register_parameter 确保其被视为权重
        this.queryTokens = register_parameter("queryTokens", randn(new long[]{numQueries, dim}));

        // 2. Cross Attention
        // 注意：LibTorch 原生 MHA 默认输入是 [TargetSeq, Batch, Dim]
        this.crossAttn = new MultiheadAttentionImpl(dim, numHeads);

        // 3. FFN
        this.ffn1 = new LinearImpl(dim, dim * 4);
        this.ffn2 = new LinearImpl(dim * 4, dim);

        // 4. 自定义 LayerNorm (1e-12 适配 Transformer)
        this.ln1 = new LayerNorm(dim, 1e-12, true);
        this.ln2 = new LayerNorm(dim, 1e-12, true);

        // 注册子模块
        register_module("crossAttn", crossAttn);
        register_module("ffn1", ffn1);
        register_module("ffn2", ffn2);
        register_module("ln1", ln1);
        register_module("ln3", ln2);
    }

    /**
     * @param x 输入节点特征 [Batch, SeqLen, Dim] 或 [BatchNodes, Dim]
     */
    public Tensor forward(Tensor x) {
        // --- 维度处理 ---
        // 如果输入是 [N, D]，视其为 [N, 1, D] 以适配批处理逻辑
        if (x.dim() == 2) {
            x = x.unsqueeze(0); // [1, N, D]
        }

        long batchSize = x.size(0);

        // 1. 准备 MultiheadAttention 的输入
        // MHA 默认要求: query [L_target, B, D], key/value [L_source, B, D]
        // x (Memory): [B, N, D] -> [N, B, D]
        Tensor mem = x.transpose(0, 1).contiguous();

        // queryTokens: [numQueries, D] -> [numQueries, B, D]
        // 必须显式 expand 以确保批次大小匹配
        Tensor tgt = queryTokens.unsqueeze(1).expand(new long[]{queryTokens.size(0), batchSize, dim}).contiguous();

        // 2. Cross Attention 核心计算
        // 返回 Tuple: {Output, Weights}
        T_TensorTensor_T mhaResult = crossAttn.forwardT_TensorTensor_T(tgt, mem, mem);
        Tensor attnOut = mhaResult.get0(); // [numQueries, B, D]

        // 3. Residual + LayerNorm (ln1)
        // 我们自定义的 LayerNorm 会处理最后一个维度的标准化
        // 传入前转回 [B, L, D] 布局是更安全的做法
        Tensor tmp = tgt.add(attnOut).transpose(0, 1).contiguous();
        Tensor h = ln1.forward(tmp); // [B, nQ, D]

        // 4. FFN: Linear -> ReLU -> Linear
        // 使用 Scalar(4.0) 这种包装并非强制，但在 Linear 的 weight 运算中要小心
        Tensor ffnHidden = relu(ffn1.forward(h));
        Tensor ffnOut = ffn2.forward(ffnHidden);

        // 5. Residual + LayerNorm (ln2)
        Tensor htmp = h.add(ffnOut);
        Tensor out = ln2.forward(htmp);

        // 返回结果 [B, numQueries, Dim]
        return out;
    }
}
/**
 * Q-Former (Querying Transformer) Component
 * 使用一组可学习的 Queries 去 "查询" 输入特征 X。
 * 结构: Cross-Attention(Q=Queries, K=X, V=X)
 */
//public class QFormer extends Module {
//    private Tensor queryTokens; // Learnable Queries [NumQueries, Dim] Parameter
//    private MultiheadAttentionImpl crossAttn;
//    private LinearImpl ffn1, ffn2;
//    private LayerNorm ln1, ln2, ln3;
//
//    public QFormer(long dim, long numHeads, long numQueries) {
//        super();
//
//        // Learnable Queries
//        // 初始化: Normal distribution
//        this.queryTokens = new Tensor(torch.randn(new long[]{numQueries, dim})); //Parameter
//        register_parameter("queryTokens", queryTokens);
//
//        // Cross Attention
//        // batch_first=true
//        this.crossAttn = new MultiheadAttentionImpl(dim, numHeads); // JavaCPP 默认可能不是 batch_first，需注意
//        // 注意：LibTorch 的 MultiheadAttentionImpl 构造函数签名可能随版本变化
//        // 这里假设标准构造，forward 时手动调整维度
//
//        // FFN
//        this.ffn1 = new LinearImpl(dim, dim * 4);
//        this.ffn2 = new LinearImpl(dim * 4, dim);
//
/// /        var opts = new LayerNormOptions();
//        this.ln1 = new LayerNorm(hiddenSize, 1e-12, true);
//        this.ln2 = new LayerNorm(hiddenSize, 1e-12, true);
////        this.ln1 = new LayerNormImpl(new LongVector(dim));
////        this.ln2 = new LayerNormImpl(new LongVector(dim));
//
//        register_module("crossAttn", crossAttn);
//        register_module("ffn1", ffn1);
//        register_module("ffn2", ffn2);
//        register_module("ln1", ln1);
//        register_module("ln2", ln2);
//    }
//
//    public Tensor forward(Tensor x) {
//        // x: [BatchNodes, Dim] (通常 org.bytedeco.pytorch.geometric.attention.QFormer 作用于 Batch 后的全图或子图)
//        // 我们假设 dimSize 1 是 Batch 维度，或者我们对全图做单一查询
//
//        // 为了适配 MHA 的输入 [Seq, Batch, Dim]，我们需要增加维度
//        // 假设 x 是单张大图 [N, D] -> [N, 1, D]
//        long N = x.size(0);
//        Tensor mem = x.unsqueeze(1);
//
//        // Queries: [NumQ, D] -> [NumQ, 1, D]
//        long nQ = queryTokens.size(0);
//        Tensor tgt = queryTokens.unsqueeze(1);
//
//        // 1. Cross Attention
//        // forward(query, key, value)
//        // Tuple: (output, weights)
//        var mhaOut = crossAttn.forward(tgt, mem, mem);
//        Tensor attnOut = mhaOut.get0(); // [NumQ, 1, D]
//
//        // Residual + Norm
//        Tensor h = ln1.forward(tgt.add(attnOut));
//
//        // 2. FFN
//        Tensor ffnOut = ffn2.forward(torch.relu(ffn1.forward(h)));
//        Tensor out = ln2.forward(h.add(ffnOut));
//
//        // Return: [NumQ, D] (压缩后的图表示)
//        return out.squeeze(1);
//    }
//}