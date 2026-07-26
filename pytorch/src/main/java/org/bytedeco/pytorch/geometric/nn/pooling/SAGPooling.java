package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * SAGPooling (Self-Attention Graph Pooling)
 * 使用 GCN 计算 Attention Score，然后保留 TopK 节点。
 */
public class SAGPooling extends TopKPooling {
    private GCNConv gnnScore;
    private long inChannels;
    private double ratio;
    public SAGPooling(long inChannels, double ratio) {
        super(inChannels, ratio); // 复用 TopKPooling 的逻辑
        // 覆盖 TopKPooling 的 weight，改用 GCN
        // GCN 输出维度为 1 (Score)
        this.inChannels = inChannels;
        this.ratio = ratio;
        this.gnnScore = new GCNConv(inChannels, 1);
        register_module("gnnScore", gnnScore);
    }

    /**
     * 重写计算 Score 的逻辑
     */
    public Tensor calculateScore(Tensor x, Tensor edge_index) {
        // TopK 使用 x @ p
        // SAG 使用 GCN(x, edge_index)
        Tensor score = gnnScore.forward(x, edge_index).tanh();
        return score.squeeze(1);
    }

    // forward 方法可以直接复用 TopKPooling 的，只要 TopKPooling 调用了 calculateScore
    // 如果 TopKPooling 没有提取该方法，则需要在这里完整重写 forward。
    // 为了演示简洁，假设 TopKPooling 已重构或此处完整实现：

    @Override
    public Tensor[] forward2(Tensor x, Tensor edge_index, Tensor batch) {
        Tensor score = calculateScore(x, edge_index);
        // ... 接下来的 Select TopK, Gate, Relabel 逻辑与 TopKPooling 完全一致 ...
        // (参考上一节 TopKPooling 代码，此处省略重复部分以节省篇幅)
        // 建议将 TopKPooling 的后半部分逻辑抽取为 protected 方法 filterAndRelabel(x, edge_index, batch, score)
        return super.topk(x, edge_index, batch); // 假设已重构
    }

    /**
     * @param x         节点特征 [N, C]
     * @param edge_index 边索引 [2, E]
     * @param batch     批次索引 [N]
     * @return Tensor[] {x_new, edge_index_new, batch_new, perm, score}
     */
    public Tensor[] sagPool(Tensor x, Tensor edge_index, Tensor batch) {
        if (batch == null) {
            batch = zeros(new long[]{x.size(0)},
                    x.options().dtype(new ScalarTypeOptional(ScalarType.Long)));
        }

        // 1. 计算 Self-Attention 分数 (使用 GNN 考虑结构信息)
        // score 形状: [N, 1] -> [N]
        Tensor score = gnnScore.forward(x, edge_index).squeeze(1);

        // 2. 选择 Top-K 节点
        long numNodes = x.size(0);
        long k = Math.max(1, (long) (numNodes * ratio));

        // 获取 Top-K 索引和分数
        T_TensorTensor_T topkRet = org.bytedeco.pytorch.global.torch.topk(score, k);
        Tensor featScores = topkRet.get0().contiguous();
        Tensor perm = topkRet.get1().contiguous();

        // 3. 特征门控与提取: x = x * tanh(score)
        // 使用激活函数后的分数作为门控信号
        Tensor gate = tanh(featScores).unsqueeze(1);
        Tensor xNew = x.index_select(0, perm).multiply(gate);
        Tensor batchNew = batch.index_select(0, perm);

        // --- 4. 建立索引映射表 (防止 JVM Crash 的关键) ---
        // 使用 scatter_ 建立 Old_Index -> New_Index 的映射
        Tensor map = full(new long[]{numNodes}, new Scalar(-1),
                edge_index.options().dtype(new ScalarTypeOptional(ScalarType.Long)));

        Tensor newIdxRange = arange(new Scalar(0), new Scalar(k),
                edge_index.options().dtype(new ScalarTypeOptional(ScalarType.Long)));

        map.scatter_(0, perm, newIdxRange);

        // --- 5. 过滤并重连边 (Filter & Relabel) ---
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        Tensor newRow = map.index_select(0, row);
        Tensor newCol = map.index_select(0, col);

        // 只有当边的两个端点都在 perm 中时，该边才保留
        Tensor mask = newRow.ge(new Scalar(0)).logical_and(newCol.ge(new Scalar(0)));

        // 提取有效边并重构 edge_index
        Tensor finalRow = newRow.masked_select(mask);
        Tensor finalCol = newCol.masked_select(mask);

        Tensor edge_indexNew = stack(new TensorVector(finalRow, finalCol), 0);

        return new Tensor[]{xNew, edge_indexNew, batchNew, perm, score};
    }
}