package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.geometric.attention.SGFormerAttention;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;
//import org.gnn.framework.attention.SGFormerAttention;
//import org.gnn.framework.layers.GCNConv; // 混合 GCN 和 Attention

public class SGFormer extends Module {
    private GCNConv gcn;
    private SGFormerAttention globalAttn;
    private LinearImpl fc;

    public SGFormer(long inChannels, long hiddenChannels, long outChannels) {
        // SGFormer 通常结合局部 GNN 和全局 Attention
        this.gcn = new GCNConv(inChannels, hiddenChannels);
        this.globalAttn = new SGFormerAttention(hiddenChannels, 1); // 1 head
        this.fc = new LinearImpl(hiddenChannels, outChannels);

        register_module("gcn", gcn);
        register_module("globalAttn", globalAttn);
        register_module("fc", fc);
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. Local GCN
        Tensor h = gcn.forward(x, edge_index).relu();

        // 2. Global Attention
        Tensor hGlobal = globalAttn.forward(h);

        // 3. Combine (Simple Add or Concat)
        h = h.add(hGlobal);

        return fc.forward(h);
    }
}