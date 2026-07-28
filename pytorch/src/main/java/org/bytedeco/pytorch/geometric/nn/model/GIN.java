package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.geometric.nn.conv.GINConv;
//import org.gnn.framework.layers.org.bytedeco.pytorch.geometric.nn.conv.GINConv; // 引用之前实现的 org.bytedeco.pytorch.geometric.nn.conv.GINConv
import java.util.ArrayList;
import java.util.List;
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
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;
/**
 * 标准 org.bytedeco.pytorch.geometric.nn.model.GIN (Graph Isomorphism Network) 模型
 * 结构: [org.bytedeco.pytorch.geometric.nn.conv.GINConv(MLP) -> BatchNorm -> ReLU] x N -> Linear(Classifier)
 */
public class GIN extends Module {

    private List<GINConv> convs;
    private List<BatchNorm1dImpl> batchNorms;
    private LinearImpl lin1, lin2;
    private double dropout;

    public GIN(long inChannels, long hiddenChannels, long outChannels, int numLayers, double dropout) {
        super();
        this.dropout = dropout;
        this.convs = new ArrayList<>();
        this.batchNorms = new ArrayList<>();

        // --- 构建 org.bytedeco.pytorch.geometric.nn.conv.GINConv 层 ---
        for (int i = 0; i < numLayers; i++) {
            long dimIn = (i == 0) ? inChannels : hiddenChannels;
            long dimOut = hiddenChannels;

            // 1. 构建该层需要的 MLP: Linear -> ReLU -> Linear
            SequentialImpl mlp = new SequentialImpl();
            mlp.push_back(new LinearImpl(dimIn, dimOut));
            mlp.push_back(new ReLUImpl());
            mlp.push_back(new LinearImpl(dimOut, dimOut));

            // 2. 构建 org.bytedeco.pytorch.geometric.nn.conv.GINConv (trainEps=true 让模型学习 epsilon)
            GINConv conv = new GINConv(mlp, true);

            // 3. 构建 BatchNorm (org.bytedeco.pytorch.geometric.nn.model.GIN 的标配)
            BatchNormOptions opt = new BatchNormOptions(dimOut);
            opt.num_features().put(dimOut);
            BatchNorm1dImpl bn = new BatchNorm1dImpl(opt);

            // 4. 保存并注册
            convs.add(conv);
            batchNorms.add(bn);

            register_module("conv_" + i, conv);
            register_module("bn_" + i, bn);
        }

        // --- 全连接分类头 ---
        this.lin1 = new LinearImpl(hiddenChannels, hiddenChannels);
        this.lin2 = new LinearImpl(hiddenChannels, outChannels);

        register_module("lin1", lin1);
        register_module("lin2", lin2);
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        Tensor h = x;

        // 循环通过每一层 org.bytedeco.pytorch.geometric.nn.model.GIN Block
        for (int i = 0; i < convs.size(); i++) {
            // 1. org.bytedeco.pytorch.geometric.nn.model.GIN Convolution
            h = convs.get(i).forward(h, edge_index);

            // 2. Batch Normalization
            h = batchNorms.get(i).forward(h);

            // 3. Activation (ReLU)
            h = torch.relu(h);

            // 4. Dropout
            h = torch.dropout(h, dropout, this.is_training());
        }

        // --- Readout / Classifier ---
        // 通常 org.bytedeco.pytorch.geometric.nn.model.GIN 会把各层的输出加起来做 Jumping Knowledge，这里简化为只用最后一层
        h = lin1.forward(h);
        h = torch.relu(h);
        h = torch.dropout(h, dropout, this.is_training());
        h = lin2.forward(h);

        return h;
    }
}
