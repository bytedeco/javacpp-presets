package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
/**
 * TopKPooling
 * 选择每个图中分数最高的 ratio * N 个节点。
 * 返回: (x, edge_index, batch, perm)
 */
public class TopKPooling extends Module {
    private long inChannels;
    private double ratio;
    private long minScore; // 或者用 multiplier

    private Tensor weight; // 投影向量 p Parameter

    public TopKPooling(long inChannels, double ratio) {
        this.inChannels = inChannels;
        this.ratio = ratio;
        this.weight = new Parameter(torch.randn(new long[]{inChannels, 1})); // new Parameter(
        register_parameter("weight", weight);
    }

    public Tensor[] topk(Tensor x, Tensor edge_index, Tensor batch) {
        if (batch == null) {
            batch = torch.zeros(new long[]{x.size(0)},
                    x.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
        }

        // 1. 计算 Score (p 是投影向量)
        Tensor norm = weight.norm(new Scalar(2)).clamp_min(new Scalar(1e-6));
        Tensor p = weight.divide(norm);
        Tensor score = x.matmul(p).squeeze(1); // [N]

        // 2. 选择 TopK 节点
        long numNodes = x.size(0);
        long k = Math.max(1, (long) (numNodes * ratio));

        // 获取 TopK 索引
        T_TensorTensor_T topkRet = torch.topk(score, k);
        Tensor featScores = topkRet.get0().contiguous();
        Tensor perm = topkRet.get1().contiguous(); // 保留的节点索引 [k]

        // 3. 特征门控与提取
        Tensor gate = torch.tanh(featScores).unsqueeze(1);
        Tensor xNew = x.index_select(0, perm).multiply(gate);
        Tensor batchNew = batch.index_select(0, perm);

        // --- 4. 修复崩溃点：重构映射表 map ---
        // 不要使用 map.index_put_(perm, range)
        Tensor map = torch.full(new long[]{numNodes}, new Scalar(-1),
                edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        // 创建 0..k-1 的新索引
        Tensor newIdxRange = torch.arange(new Scalar(0), new Scalar(k),
                edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        // 使用 scatter_ 替代 index_put_，这是最安全的指针操作方式
        // map[perm] = newIdxRange
        map.scatter_(0, perm, newIdxRange);

        // 5. 过滤边 (Filter & Relabel)
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        // 映射旧节点到新索引
        Tensor newRow = map.index_select(0, row);
        Tensor newCol = map.index_select(0, col);

        // 掩码：两端都在新集合中 (newRow >= 0 AND newCol >= 0)
        // 使用 Scalar(0) 显式包装
        Tensor mask = newRow.ge(new Scalar(0)).logical_and(newCol.ge(new Scalar(0)));

        // 提取有效边 (直接使用 mask 进行索引，更安全)
        Tensor finalRow = newRow.masked_select(mask);
        Tensor finalCol = newCol.masked_select(mask);

        // 拼接成新的 edge_index [2, E_new]
        Tensor edge_indexNew = torch.stack(new TensorVector(finalRow, finalCol), 0);

        return new Tensor[]{xNew, edge_indexNew, batchNew, perm, score};
    }

    /**
     * @return Tensor[] {x, edge_index, batch, perm, score}
     */
    public Tensor[] forward2(Tensor x, Tensor edge_index, Tensor batch) {
        if (batch == null) {
            batch = torch.zeros(new long[]{x.size(0)}, x.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
        }

        // 1. Calculate Score: score = x @ p / ||p||
        Tensor norm = weight.norm(new Scalar(2)).clamp_min(new Scalar(1e-6));
        Tensor p = weight.div(norm);
        Tensor score = x.matmul(p).squeeze(1); // [N]

        // 2. Select TopK Nodes per graph
        // 为了避免复杂的 segmented_sort，我们使用一个技巧：
        // 如果我们只做 mask 而不强制 batch 比例，可以用全局 topk。
        // 但标准的 TopKPooling 是每个图保留 k 个。
        // 在 LibTorch 中实现 per-graph topk 比较繁琐，这里实现一个 Mask 方式：
        // score_soft = tanh(score)

        // 我们这里实现简化版：Global TopK (假设图大小差不多) 或者 Mask based on threshold.
        // 为了精确复现，我们需要 sort batch.
        // 这里采用 Masking 策略：对 score 进行 sigmoid，保留 > threshold? 不，TopK 是硬保留。

        // --- 核心：Per-Graph TopK ---
        long numNodes = x.size(0);
        long k = (long) (numNodes * ratio);
        k = Math.max(1, k); // 至少保留1个

        // 全局 TopK (简化实现，工业界通常用 segmented sort)
        // sort returns (values, indices)
        T_TensorTensor_T topkRet = torch.topk(score, k);
        Tensor perm = topkRet.get1(); // 保留的节点索引 [k]
        Tensor featScores = topkRet.get0(); // [k]

        // 如果要严格按 Graph 比例，需要循环或 sort batch。
        // 鉴于 Java 循环慢，这里假设 global ratio 近似等于 local ratio。

        // 3. Gate Feature: x = x * tanh(score)
        Tensor gate = torch.tanh(featScores).unsqueeze(1);
        Tensor xNew = x.index_select(0, perm).mul(gate);
        Tensor batchNew = batch.index_select(0, perm);

        // 4. Filter Edge Index & Relabel
        // 这是最难的一步：重新映射索引 
        // Old Index -> New Index (0 to k-1)

        // 创建映射表 map: [N] filled with -1
        Tensor map = torch.full(new long[]{numNodes}, new Scalar(-1), edge_index.options());
        // map[perm] = 0..k-1
        Tensor newIdxRange = torch.arange(new Scalar(k), edge_index.options());
        map.index_put_(new TensorIndexVector(perm), newIdxRange);

        // 获取边的端点
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        // 映射端点
        Tensor newRow = map.index_select(0, row);
        Tensor newCol = map.index_select(0, col);

        // 保留两个端点都在 perm 中的边 (newRow >= 0 AND newCol >= 0)
        Tensor mask = newRow.ge(new Scalar(0)).logical_and(newCol.ge(new Scalar(0)));

        // 提取有效边
        Tensor validIndices = mask.nonzero().squeeze(1);
        Tensor finalRow = newRow.index_select(0, validIndices);
        Tensor finalCol = newCol.index_select(0, validIndices);

        Tensor edge_indexNew = torch.stack(new TensorVector(finalRow, finalCol), 0);

        return new Tensor[]{xNew, edge_indexNew, batchNew, perm, score};
    }
}