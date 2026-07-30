package samples.demo.layer;

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

    public static void main(String[] args) {
        long numNodes = 5, feats = 4, classes = 2, hidden = 8;
        Tensor x = torch.randn(numNodes, feats);
        Tensor edgeIndex = torch.tensor(new long[]{0,1,1,2,2,3,3,4, 1,0,2,1,3,2,4,3})
                .reshape(2, 8);
        GraphData data = new GraphData(x, edgeIndex);
        SimpleGCN model = new SimpleGCN(feats, classes, hidden);
        model.train(true);
        Tensor out = model.forward(data);
        System.out.println("SimpleGCN out shape: [" + out.size(0) + ", " + out.size(1) + "]");
        if (out.size(0) != numNodes || out.size(1) != classes) {
            throw new RuntimeException("shape mismatch");
        }
        System.out.println("✅ SimpleGCN OK");
    }

}
