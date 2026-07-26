package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;
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

public class GAE extends org.bytedeco.pytorch.nn.Module {
    protected Module encoder;
    protected InnerProductDecoder decoder;

    public GAE(Module encoder) {
        this(encoder, new InnerProductDecoder());
    }

    public GAE(Module encoder, InnerProductDecoder decoder) {
        super();
        this.encoder = encoder;
        this.decoder = decoder;
        register_module("encoder", encoder);
        register_module("decoder", decoder);
    }

    public Tensor encode(Tensor x, Tensor edge_index) {
        // 调用编码器（通常是 GCNConv 或 PNAConv）
        // 假设 encoder 有 forward(Tensor x, Tensor edge_index)
        return ((GCNConv)encoder).forward(x, edge_index);
    }

    public Tensor decode(Tensor z, Tensor edge_index, boolean sigmoid) {
        return decoder.forward(z, edge_index, sigmoid);
    }

    /**
     * 一比一还原 PyG recon_loss 逻辑
     */
    public Tensor recon_loss(Tensor z, Tensor posedge_index, Tensor negedge_index) {
        double EPS = 1e-15;

        // 正样本损失: -log(sigmoid(dot(z_i, z_j)))
        Tensor posPred = decode(z, posedge_index, true);
        Tensor posLoss = posPred.add(new Scalar((float) EPS)).log().mean().neg();

        // 负样本损失: -log(1 - sigmoid(dot(z_i, z_j)))
        // 如果没有传入 negedge_index，某些版本会进行随机负采样
        Tensor negPred = decode(z, negedge_index, true);
        Tensor negLoss = torch.ones_like(negPred).subtract(negPred).add(new Scalar((float) EPS)).log().mean().neg();

        return posLoss.add(negLoss);
    }
}

// org.bytedeco.pytorch.geometric.nn.model.GAE
//public class GAE extends org.bytedeco.pytorch.nn.Module {
//    protected org.bytedeco.pytorch.nn.Module encoder;
//    protected InnerProductDecoder decoder;
//
//    public GAE(Module encoder) {
//        this.encoder = encoder;
//        this.decoder = new InnerProductDecoder();
//        register_module("encoder", encoder);
//    }
//
//    public Tensor encode(Tensor x, Tensor edge_index) {
//        // Cast or call specific method
//        // Assume encoder returns Z
//        // In Java, we might need an Interface 'Encoder'
//        return ((GCNConv)encoder).forward(x, edge_index);
//    }
//
//    public Tensor decode(Tensor z, Tensor edge_index) {
//        return decoder.forward(z, edge_index, true);
//    }
//
//    public Tensor reconLoss(Tensor z, Tensor posedge_index, Tensor negedge_index) {
//        // BCE Loss logic
//        return torch.tensor(0f); // Placeholder
//    }
//}