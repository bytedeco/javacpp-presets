package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;

public class GatedGraphConv extends MessagePassing {
    private GRUCellImpl gru;
    private int numLayers; // 循环次数

    public GatedGraphConv(long outChannels, int numLayers) {
        super("add");
        this.numLayers = numLayers;
        // 输入是聚合后的特征 (outChannels)，隐状态也是 outChannels
        this.gru = new GRUCellImpl(outChannels, outChannels);
        register_module("gru", gru);
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        Tensor h = x;

        // 循环 t 次
        for (int i = 0; i < numLayers; i++) {
            // 1. 聚合邻居: m = A * h
            Tensor m = propagate(edge_index, h);

            // 2. GRU 更新: h_new = GRU(input=m, hidden=h)
            h = gru.forward(m, h);
        }
        return h;
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