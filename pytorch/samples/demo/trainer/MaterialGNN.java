package samples.demo.trainer;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.nn.conv.GINConv;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConvV2;
import org.bytedeco.pytorch.geometric.nn.conv.TransformerConv;
import org.bytedeco.pytorch.geometric.nn.norm.InstanceNorm;
import org.bytedeco.pytorch.geometric.nn.norm.LayerNorm;
import org.bytedeco.pytorch.geometric.nn.pooling.GlobalPooling;

import static org.bytedeco.pytorch.global.torch.*;

public class MaterialGNN extends org.bytedeco.pytorch.nn.Module {
    private final GINConv gin;
    private final SAGEConvV2 sage;
    private final TransformerConv trans;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final LayerNorm norm3;
    private final LinearImpl fc1;
    private final LinearImpl out;
    private final EmbeddingImpl elementEmbed;
    private InstanceNorm bn;

    public MaterialGNN(long numElements, long embedDim, long hiddenDim) {
        super();
        this.elementEmbed = new EmbeddingImpl(numElements, embedDim);
        // 1. GIN 层：捕获精确的化学键局部结构 (适合识别晶格)
        SequentialImpl mlp = new SequentialImpl();
        mlp.push_back(new LinearImpl(embedDim, hiddenDim));
        mlp.push_back(new ReLUImpl());
        this.gin = new GINConv(mlp, true);
        this.norm1 = new LayerNorm(hiddenDim, 1e-5, true);

        // 2. SAGE 层：模拟材料内部的电荷/力场分布平滑
        this.sage = new SAGEConvV2(hiddenDim,hiddenDim, hiddenDim);
        this.norm2 = new LayerNorm(hiddenDim, 1e-5, true);

        // 3. Transformer 层：模拟非键合的长程相互作用 (如范德华力)
        this.trans = new TransformerConv(hiddenDim, hiddenDim, 2);
        this.norm3 = new LayerNorm(hiddenDim * 2, 1e-5, true);

        // 4. 预测头
        this.fc1 = new LinearImpl(hiddenDim * 2, hiddenDim);
//        this.bn = new InstanceNorm(hiddenDim*2, 1e-5, 0.1, true, true);
        this.out = new LinearImpl(hiddenDim, 1);

        register_module("elementEmbed", elementEmbed);
        register_module("gin", gin);
        register_module("norm1", norm1);
        register_module("sage", sage);
        register_module("norm2", norm2);
        register_module("trans", trans);
        register_module("norm3", norm3);
//        register_module("bn", bn);
        register_module("fc1", fc1);
        register_module("platform/out", out);

    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        x = elementEmbed.forward(x.to(kLong()));
        if (x.dim() == 3) x = x.squeeze(1); // 维度校正

        // 核心 GNN 链路
        // 第一阶段：结构特征提取
        x = relu(gin.forward(x, edge_index));
        x = norm1.forward(x);

        // 第二阶段：邻域场信息交换
        x = relu(sage.forward(x, edge_index));
        x = norm2.forward(x);

        // 第三阶段：全局关联建模
        x = relu(trans.forward(x, edge_index));
        x = norm3.forward(x);

        // 第四阶段：Readout (Global Pooling)
        Tensor batch = zeros(new long[]{x.size(0)}, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).to(x.device(), ScalarType.Long);
//        x = bn.forward(x, batch);
        x = GlobalPooling.pool(x, batch, "mean"); // 聚合全材料特征

        x = relu(fc1.forward(x));
        return out.forward(x);
    }
}