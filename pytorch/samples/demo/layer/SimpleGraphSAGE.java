package samples.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConv; // 引入你实现的 SAGE 卷积层
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * 使用 torch-geometric (Java版) 实现的 GraphSAGE 模型
 */
public class SimpleGraphSAGE extends Module {
    private SAGEConv conv1;
    private SAGEConv conv2;

    public SimpleGraphSAGE(long numNodeFeatures, long numClasses, long hiddenChannels) {
        // 1. 初始化 SAGEConv。默认聚合器通常在构造函数中指定，如 "mean"
        // 注册子模块确保参数能被 optimizer 追踪
        this.conv1 = register_module("conv1", new SAGEConv(numNodeFeatures, hiddenChannels));
        this.conv2 = register_module("conv2", new SAGEConv(hiddenChannels, numClasses));
    }

    /**
     * 前向传播
     * @param data 包含 x (节点特征) 和 edgeIndex (邻接关系)
     */
    public Tensor forward(GraphData data) {
        Tensor x = data.x;
        Tensor edgeIndex = data.edge_index;

        // 第一层 SAGE 卷积 + 激活
        x = conv1.forward(x, edgeIndex);
        x = relu(x);

        // 训练模式下的 Dropout 处理
        x = dropout(x, 0.5, is_training());

        // 第二层 SAGE 卷积
        x = conv2.forward(x, edgeIndex);

        // 返回 LogSoftmax 结果用于分类
        return torch.log_softmax(x, 1);
    }

    public static void main(String[] args) {
        long numNodes = 5, feats = 4, classes = 2, hidden = 8;
        Tensor x = torch.randn(numNodes, feats);
        Tensor edgeIndex = torch.tensor(new long[]{0,1,1,2,2,3,3,4, 1,0,2,1,3,2,4,3}).reshape(2, 8);
        GraphData data = new GraphData(x, edgeIndex);
        SimpleGraphSAGE model = new SimpleGraphSAGE(feats, classes, hidden);
        Tensor out = model.forward(data);
        System.out.println("SimpleGraphSAGE out shape: [" + out.size(0) + ", " + out.size(1) + "]");
        System.out.println("✅ SimpleGraphSAGE OK");
    }

}
