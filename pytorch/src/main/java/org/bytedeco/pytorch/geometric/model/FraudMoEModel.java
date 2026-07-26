package org.bytedeco.pytorch.geometric.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

/**
 * 基于 MoE 和 org.bytedeco.pytorch.geometric.nn.conv.TransformerConv 的反欺诈模型
 */
public class FraudMoEModel extends Module {

    private MoETransformerLayer moeLayer;
    private LinearImpl classifier;
    private double dropoutRate;

    public FraudMoEModel(long inChannels, long hiddenChannels, long outChannels,
                         long heads, int numExperts, double dropout) {
        super();
        this.dropoutRate = dropout;

        // MoE Layer: 输入维度 -> 隐层维度 (实际输出宽度是 hidden * heads)
        // 注意：org.bytedeco.pytorch.geometric.nn.conv.TransformerConv 的输出维度是 heads * outChannels
        this.moeLayer = new MoETransformerLayer(inChannels, hiddenChannels, heads, numExperts);

        long moeOutputDim = hiddenChannels * heads;

        // Classifier: MoE输出 -> 最终 Logits
        this.classifier = new LinearImpl(moeOutputDim, outChannels);

        register_module("moeLayer", moeLayer);
        register_module("classifier", classifier);
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. MoE Layer
        Tensor h = moeLayer.forward(x, edge_index);

        // 2. Activation
        h = torch.relu(h);

        // 3. Dropout
        h = torch.dropout(h, dropoutRate, this.is_training());

        // 4. Classifier
        return classifier.forward(h);
    }
}
