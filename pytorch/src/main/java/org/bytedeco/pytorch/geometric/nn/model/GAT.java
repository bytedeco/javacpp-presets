package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GATConv;
//import org.gnn.framework.layers.org.bytedeco.pytorch.geometric.nn.conv.GATConv; // 引用你之前实现的 org.bytedeco.pytorch.geometric.nn.conv.GATConv


/**
 * org.bytedeco.pytorch.geometric.nn.model.GAT (Graph Attention Network) 完整模型
 * 结构: Input -> [org.bytedeco.pytorch.geometric.nn.conv.GATConv(Concat) + ELU + Dropout] -> [org.bytedeco.pytorch.geometric.nn.conv.GATConv(Mean)] -> Output
 */
public class GAT extends Module {

    private GATConv conv1;
    private GATConv conv2;

    private double dropoutRate;
    private long outHeads;     // 输出层的头数
    private long outChannels;  // 输出层的特征数 (num_classes)
    private double negativeSlope = 0.2;
    /**
     * @param inChannels 输入特征维度
     * @param hiddenChannels 隐层特征维度
     * @param outChannels 输出类别数
     * @param heads 隐层的注意力头数 (Layer 1)
     * @param outHeads 输出层的注意力头数 (Layer 2)
     * @param dropout Dropout 概率
     */
    public GAT(long inChannels, long hiddenChannels, long outChannels,
               long heads, long outHeads, double dropout) {
        super();
        this.dropoutRate = dropout;
        this.outHeads = outHeads;
        this.outChannels = outChannels;

        // Layer 1: 输入层 -> 隐层
        // org.bytedeco.pytorch.geometric.nn.conv.GATConv 默认行为是 Concat，所以输出维度是 heads * hiddenChannels
        this.conv1 = new GATConv(inChannels, hiddenChannels, heads, negativeSlope);

        // Layer 2: 隐层 -> 输出层
        // 输入维度必须匹配 Layer 1 的输出: heads * hiddenChannels
        // 我们先让 org.bytedeco.pytorch.geometric.nn.conv.GATConv 输出 outHeads * outChannels (Concat)，然后在 forward 里手动做 Mean
        this.conv2 = new GATConv(hiddenChannels * heads, outChannels, outHeads, negativeSlope);

        register_module("conv1", conv1);
        register_module("conv2", conv2);
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        // --- Layer 1 ---
        // 1. Dropout on Features (根据论文，输入特征也要 Dropout)
        Tensor h = torch.dropout(x, dropoutRate, this.is_training());

        // 2. org.bytedeco.pytorch.geometric.nn.model.GAT Conv 1
        h = conv1.forward(h, edge_index);

        // 3. ELU Activation (org.bytedeco.pytorch.geometric.nn.model.GAT 论文标配是 ELU 而不是 ReLU)
        h = torch.elu(h);

        // --- Layer 2 ---
        // 4. Dropout on Hidden Features
        h = torch.dropout(h, dropoutRate, this.is_training());

        // 5. org.bytedeco.pytorch.geometric.nn.model.GAT Conv 2
        h = conv2.forward(h, edge_index);
        // 此时 h 的形状是 [N, outHeads * outChannels]

        // 6. Output Averaging (org.bytedeco.pytorch.geometric.nn.model.GAT 输出层通常取平均，而不是拼接)
        if (outHeads > 1) {
            long numNodes = h.size(0);
            // Reshape: [N, outHeads * outChannels] -> [N, outHeads, outChannels]
            h = h.view(numNodes, outHeads, outChannels);

            // Mean over heads dimension (dim 1)
            // [N, outHeads, outChannels] -> [N, outChannels]
            h = h.mean(new long[]{1}, false, new ScalarTypeOptional(h.scalar_type()));
        }

        // 7. Log Softmax (通常用于分类任务，或者直接输出 Logits)
        return h;
    }
}
