package org.bytedeco.pytorch.geometric.model;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConv;

import static org.bytedeco.pytorch.global.torch.*;

public class GraphSAGEModel extends Module {
    private final SAGEConv conv1, conv2;

    public GraphSAGEModel(long inChannels, long hiddenChannels) {
        this.conv1 = new SAGEConv(inChannels, hiddenChannels);
        this.conv2 = new SAGEConv(hiddenChannels, hiddenChannels);
        register_module("conv1", conv1);
        register_module("conv2", conv2);
    }

    public Tensor forward(Tensor x, Tensor edgeIndex) {
        x = relu(conv1.forward(x, edgeIndex));
        x = conv2.forward(x, edgeIndex);
        return x;
    }

    // 计算预测分数：基于节点嵌入的点积
    public Tensor score(Tensor z, Tensor edgeIndex) {
//        try (PointerScope scope = new PointerScope()) {
            // 1. 维度防御检查
            if (z.size(0) == 0) {
                throw new RuntimeException("Embedding tensor 'z' is empty!");
            }
         
            // 2. 提取源节点和目标节点索引
            Tensor srcIdx = edgeIndex.select(0, 0);
            Tensor dstIdx = edgeIndex.select(0, 1);

            // 3. 越界防御检查 (非常重要！)
            long maxIdx = Math.max(srcIdx.max().item().toLong(), dstIdx.max().item().toLong());
            if (maxIdx >= z.size(0)) {
                throw new RuntimeException(String.format(
                        "Index out of bounds: Max edge index %d >= Embedding size %d. " +
                                "Did you forget to reindex your edges or use full-size embeddings?",
                        maxIdx, z.size(0)));
            }

            // 4. 计算得分
            Tensor src = z.index_select(0, srcIdx.to(ScalarType.Int));
            Tensor dst = z.index_select(0, dstIdx.to(ScalarType.Int));

            // 返回点积结果并 detach
            return (src.mul(dst)).sum(1);//.detach();
//        }
    }

    public Tensor score2(Tensor z, Tensor edgeIndex) {
        Tensor src = z.index_select(0, edgeIndex.select(0, 0).to(ScalarType.Int));
        Tensor dst = z.index_select(0, edgeIndex.select(0, 1).to(ScalarType.Int));
        return (src.mul(dst)).sum(1);
    }
}