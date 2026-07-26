package org.bytedeco.pytorch.geometric.model;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConv;
//import org.gnn.framework.layers.org.bytedeco.pytorch.geometric.nn.conv.SAGEConv; // 导入你之前实现的层


/**
 * 金融反欺诈 GNN 模型
 * 架构: SAGE -> ReLU -> Dropout -> SAGE -> ReLU -> Linear
 */
public class FraudGNNModel extends Module {

    private SAGEConv conv1;
    private SAGEConv conv2;
    private LinearImpl classifier; // 最终分类器
    private double dropoutRate;

    public FraudGNNModel(long inChannels, long hiddenChannels, long outChannels, double dropout) {
        super();
        this.dropoutRate = dropout;

        // 第一层图卷积: [In -> Hidden]
        this.conv1 = new SAGEConv(inChannels, hiddenChannels);

        // 第二层图卷积: [Hidden -> Hidden]
        this.conv2 = new SAGEConv(hiddenChannels, hiddenChannels);

        // 全连接分类层: [Hidden -> 1] (二分类输出一个 Logit)
        this.classifier = new LinearImpl(hiddenChannels, outChannels);

        // 注册模块
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("classifier", classifier);
    }

    /**
     * 前向传播
     * @param x 节点特征 [N, F]
     * @param edge_index 交易图结构 [2, E]
     * @return Logits [N, 1]
     */
    public Tensor forward(Tensor x, Tensor edge_index) {
        // --- Layer 1 ---
        Tensor h = conv1.forward(x, edge_index);
        h = torch.relu(h);
        // Dropout (注意: JavaCPP 中 dropout 是 functional 的，需要传 training 状态)
        h = torch.dropout(h, dropoutRate, this.is_training());

        // --- Layer 2 ---
        h = conv2.forward(h, edge_index);
        h = torch.relu(h);
        h = torch.dropout(h, dropoutRate, this.is_training());

        // --- Classifier ---
        // 输出 Logits，配合 BCEWithLogitsLoss 使用更加数值稳定
        return classifier.forward(h);
    }
}