package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.norm.BatchNorm;

public class DeepGCNLayer extends Module {
    private Module conv;
    private Module norm;
    private Module act;
    private Module block; // Usually 'res+' or 'dense'
    private double dropout;

    public DeepGCNLayer(Module conv, Module norm, Module act, double dropout) {
        this.conv = conv;
        this.norm = norm;
        this.act = act; // e.g., ReLU
        this.dropout = dropout;

        register_module("conv", conv);
        if(norm!=null) register_module("norm", norm);
        if(act!=null) register_module("act", act);
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        Tensor h = x;

        // 1. Pre-activation/norm style: Norm -> Act -> Conv
        if (norm != null) h = ((BatchNorm)norm).forward(h); // Cast needed
        if (act != null) h = torch.relu(h); // Simplified
        h = torch.dropout(h, dropout, is_training());

        // Conv
        // h = conv(h, edge_index)
        // 此处需要统一接口，假设 conv 有 forward(x, edge_index)
        // h = conv.forward(h, edge_index);

        // 2. Residual Connection
        // x + h
        return x.add(h);
    }
}