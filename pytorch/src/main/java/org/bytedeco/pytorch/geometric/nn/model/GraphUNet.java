package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.*;
//import org.gnn.framework.layers.org.bytedeco.pytorch.geometric.nn.conv.GCNConv;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.geometric.nn.pooling.TopKPooling;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;

public class GraphUNet extends Module {
    private GCNConv downConv1, downConv2;
    private TopKPooling pool1;
    private GCNConv upConv1;

    public GraphUNet(long inChan, long hiddenChan, long outChan, double poolRatio) {
        downConv1 = new GCNConv(inChan, hiddenChan);
        pool1 = new TopKPooling(hiddenChan, poolRatio);
        downConv2 = new GCNConv(hiddenChan, hiddenChan);
        upConv1 = new GCNConv(hiddenChan * 2, outChan); // Concat skip connection

        register_module("downConv1", downConv1);
        register_module("pool1", pool1);
        register_module("downConv2", downConv2);
        register_module("upConv1", upConv1);
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. Down
        x = downConv1.forward(x, edge_index).relu();
        Tensor x1 = x; // Skip connection

        Tensor[] poolRet = pool1.topk(x, edge_index, (Tensor)null);
        Tensor xPool = poolRet[0];
        Tensor edgePool = poolRet[1];
        Tensor perm = poolRet[3]; // Indices for unpooling

        // 2. Bottom
        xPool = downConv2.forward(xPool, edgePool).relu();

        // 3. Unpool: scatter pooled features back into full node set via perm.
        // index_copy_(dim, index[K], source[K,C]) is robust under JavaCPP;
        // index_put_(TensorIndexVector(perm), …) SIGSEGVs with 1-D Long perm.
        Tensor xUp = torch.zeros_like(x1); // [N, hidden]
        xUp.index_copy_(0, perm, xPool);

        // Skip connection (concat)
        xUp = torch.cat(new TensorVector(xUp, x1), 1); // [N, 2*hidden]

        // 4. Output on original topology
        return upConv1.forward(xUp, edge_index);
    }
}