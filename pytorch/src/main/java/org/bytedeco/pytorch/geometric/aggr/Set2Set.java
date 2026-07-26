package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.c10.*;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;


/**
 * org.bytedeco.pytorch.geometric.aggr.Set2Set org.bytedeco.pytorch.geometric.aggr.Aggregation
 * 基于 LSTM + Attention 的迭代式聚合。
 * 输入: 节点特征集合
 * 输出: [BatchSize, 2 * InChannels] (Query + Readout)
 */
public class Set2Set extends Aggregation {

    private long inChannels;
    private long processingSteps;
    private long numLayers;

    private LSTMCellImpl lstm;
    private LinearImpl scr; // 用于计算 Attention Score 的线性层

    /**
     * @param inChannels 输入特征维度
     * @param processingSteps 迭代步数 (通常 3-10)
     * @param numLayers LSTM 层数 (本实现简化为 1 层，如需多层需堆叠 LSTMCell)
     */
    public Set2Set(long inChannels, long processingSteps, long numLayers) {
        this.inChannels = inChannels;
        this.processingSteps = processingSteps;
        this.numLayers = numLayers;

        // LSTM 输入: [q_t, r_t] -> 维度 2 * inChannels
        // LSTM 隐层: inChannels
        this.lstm = new LSTMCellImpl(2 * inChannels, inChannels);

        // Attention Score 计算层
        // 输入: [NodeFeat || Query] -> 2 * inChannels
        // 输出: Scalar Score
        this.scr = new LinearImpl(2 * inChannels, 1);

        register_module("lstm", lstm);
        register_module("scr", scr);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        long batchSize = dimSize;

        // 1. 初始化 Hidden State (H) and Cell State (C)
        // org.bytedeco.pytorch.geometric.aggr.Set2Set 中，Query (q) 通常就是 LSTM 的 hidden state
        // 形状: [Batch, InChannels]
        Tensor q = torch.zeros(new long[]{batchSize, inChannels}, x.options());
        Tensor c = torch.zeros(new long[]{batchSize, inChannels}, x.options());

        // 定义 Readout 变量 r (初始化为0)
        Tensor r = torch.zeros(new long[]{batchSize, inChannels}, x.options());

        // 2. 迭代处理 Processing Steps
        for (int i = 0; i < processingSteps; i++) {

            // --- A. 生成 Attention Scores ---

            // 将 q (Graph Level) 广播回 x (Node Level)
            // q: [Batch, In] -> q_expanded: [N, In]
            Tensor qExpanded = q.index_select(0, index);

            // 拼接 Node Features 和 Query: [N, 2*In]
            // 注意：JavaCPP 需要用 TensorVector
            Tensor catInput = torch.cat(new TensorVector(x, qExpanded), 1);

            // 计算分数: [N, 1]
            Tensor scores = scr.forward(catInput);

            // --- B. 计算 Attention Weights (Softmax) ---
            // alpha: [N, 1]
            // 使用我们之前实现的 org.bytedeco.pytorch.geometric.utils.AggrUtils.scatter_softmax
            // 注意 squeeze/unsqueeze 处理维度匹配
            Tensor alpha = AggrUtils.scatter_softmax(scores.squeeze(1), index, dimSize).unsqueeze(1);

            // --- C. 计算 Readout (Weighted Sum) ---
            // r = Sum( alpha * x )
            // [N, In] * [N, 1] -> [N, In]
            Tensor weighted = x.mul(alpha);
            r = AggrUtils.scatter(weighted, index, dimSize, "sum");

            // --- D. LSTM Update ---
            // Input to LSTM: [q, r] -> [Batch, 2*In]
            Tensor lstmInput = torch.cat(new TensorVector(q, r), 1);

            // LSTMCell forward 返回 Tuple: (h_new, c_new)
            // 在 JavaCPP 中，我们需要手动解包
            T_TensorTensor_T lstmState = new T_TensorTensor_T(q,c);
            T_TensorTensor_T hiddenTuple = lstm.forward(lstmInput, lstmState);

            // 更新 q (即 hidden state) 和 c
            // 注意：必须复制引用或 clone，否则下一轮循环可能会有问题
            q = hiddenTuple.get0();
            c = hiddenTuple.get1();
        }

        // 3. 最终输出: Concat([q, r])
        // 输出维度: [Batch, 2 * InChannels]
        return torch.cat(new TensorVector(q, r), 1);
    }
}