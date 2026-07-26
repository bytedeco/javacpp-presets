package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;

public class GroupAddRev extends Module {
    private Module F; // 第一个 GNN 块
    private Module G; // 第二个 GNN 块
    private double dropout;

    public GroupAddRev(Module F, Module G, double dropout) {
        this.F = F;
        this.G = G;
        this.dropout = dropout;
        register_module("F", F);
        register_module("G", G);
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. Split Channels into 2 groups
        long channels = x.size(1);
        long c2 = channels / 2;

        // split returns a vector of tensors
        TensorVector chunks = torch.split(x, c2, 1);
        Tensor x1 = chunks.get(0);
        Tensor x2 = chunks.get(1);

        // 2. Reversible Update Steps
        // Step 1: x1 = x1 + F(x2)
        // cast needed if F is generic Module
        Tensor outF = ((GCNConv)F).forward(x2, edge_index);
        outF = torch.dropout(outF, dropout, is_training());
        x1 = x1.add(outF);

        // Step 2: x2 = x2 + G(x1)
        Tensor outG = ((GCNConv)G).forward(x1, edge_index);
        outG = torch.dropout(outG, dropout, is_training());
        x2 = x2.add(outG);

        // 3. Concat
        return torch.cat(new TensorVector(x1, x2), 1);
    }

    // 注意：真正的内存节省需要在 C++ 层实现 backward 时的重计算逻辑。
    // 在纯 Java/LibTorch 前向调用中，PyTorch 自动微分引擎默认仍会缓存图。
    // 这里的实现复现了模型结构，适用于推理或标准训练。
}