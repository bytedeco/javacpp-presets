package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
//import org.gnn.framework.nn.org.bytedeco.pytorch.geometric.nn.conv.MessagePassing;

public class SGConv extends MessagePassing {
    private LinearImpl lin;
    private int K;

    public SGConv(long inChannels, long outChannels, int K) {
        super("add");
        this.K = K;
        this.lin = new LinearImpl(inChannels, outChannels);
        register_module("lin", lin);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        Tensor xRun = x;
        // 连续传播 K 次，中间没有非线性变换
        for (int i = 0; i < K; i++) {
            xRun = propagate(edge_index, xRun);
        }
        // 最后只做一次线性变换
        return lin.forward(xRun);
    }
    /**
     * 必须匹配基类签名：(x_j, x_i, edge_index, edge_attr)
     * 哪怕 SAGE 只需要 x_j，参数也必须写全！
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // GraphSAGE 的 message 就是邻居特征本身
        // 如果以后要支持带权重的 SAGE，可以在这里处理 edge_attr
        return x_j;
    }
}