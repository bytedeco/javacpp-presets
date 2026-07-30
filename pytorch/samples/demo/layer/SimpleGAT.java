package samples.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.nn.conv.GATConv; 
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * 使用 torch-geometric (Java版) 实现的 GAT 模型
 */
public class SimpleGAT extends Module {
    private GATConv conv1;
    private GATConv conv2;

    public SimpleGAT(long numNodeFeatures, long numClasses, long hiddenChannels, int heads) {
        // 第一层：多头注意力，输出维度拼接 (concat=true)
        // 注册子模块
        this.conv1 = register_module("conv1",
                new GATConv(numNodeFeatures, hiddenChannels, heads, true, 0.6));

        // 第二层：输出层，不拼接而取平均 (concat=false)，将维度从 hiddenChannels * heads 降至 numClasses
        this.conv2 = register_module("conv2",
                new GATConv(hiddenChannels * heads, numClasses, 1, false, 0.6));
    }

    /**
     * 前向传播
     */
    public Tensor forward(GraphData data) {
        Tensor x = data.x;
        Tensor edgeIndex = data.edge_index;

        // 1. 输入 Dropout
        x = torch.dropout(x, 0.6, is_training());

        // 2. 第一层卷积 + ELU 激活 (GAT 常用 ELU 替代 ReLU)
        x = conv1.forward(x, edgeIndex);
        x = torch.elu(x);

        // 3. 中间层 Dropout
        x =torch.dropout(x, 0.6, is_training());

        // 4. 第二层卷积
        x = conv2.forward(x, edgeIndex);

        // 5. 分类输出
        return torch.log_softmax(x, 1);
    }

    public static void main(String[] args) {
        long numNodes = 5, feats = 4, classes = 2, hidden = 8;
        Tensor x = torch.randn(numNodes, feats);
        Tensor edgeIndex = torch.tensor(new long[]{0,1,1,2,2,3,3,4, 1,0,2,1,3,2,4,3}).reshape(2, 8);
        GraphData data = new GraphData(x, edgeIndex);
        SimpleGAT model = new SimpleGAT(feats, classes, hidden, 2);
        Tensor out = model.forward(data);
        System.out.println("SimpleGAT out shape: [" + out.size(0) + ", " + out.size(1) + "]");
        System.out.println("✅ SimpleGAT OK");
    }

}
