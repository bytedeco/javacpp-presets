package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 使用 torch-geometric 实现的简单 GCN 模型
 */
public class SimpleGCN extends Module {
    // 声明层：GCNConv 是你自定义或框架提供的 GNN 层
    private GCNConv conv1;
    private GCNConv conv2;

    public SimpleGCN(long numNodeFeatures, long numClasses, long hiddenChannels) {
        // 初始化层并注册子模块，以便 optimizer 能找到参数
        this.conv1 = register_module("conv1", new GCNConv(numNodeFeatures, hiddenChannels));
        this.conv2 = register_module("conv2", new GCNConv(hiddenChannels, numClasses));
    }

    /**
     * 前向传播
     * @param data 包含 x (features) 和 edgeIndex 的图数据对象
     */
    public Tensor forward(GraphData data) {
        Tensor x = data.x;
        Tensor edgeIndex = data.edge_index;

        // 第一层卷积 + ReLU
        x = conv1.forward(x, edgeIndex);
        x = relu(x);

        // Dropout 层：注意训练模式判断
        x = dropout(x, 0.5, is_training());

        // 第二层卷积
        x = conv2.forward(x, edgeIndex);

        // 节点分类通常使用 log_softmax
        return log_softmax(x, 1);
    }
}