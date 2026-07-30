package samples.demo.trainer;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.nn.conv.GATConv;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConv;
import org.bytedeco.pytorch.geometric.nn.conv.TransformerConv;
import org.bytedeco.pytorch.geometric.nn.norm.LayerNorm;
import org.bytedeco.pytorch.geometric.nn.pooling.GlobalPooling;

import static org.bytedeco.pytorch.global.torch.*;

public class ProteinGNN extends org.bytedeco.pytorch.nn.Module {
    private final EmbeddingImpl embed;
    private final SAGEConv sage;
    private final GATConv gat;
    private final LayerNorm norm1;
    private final TransformerConv trans;
    private final LayerNorm norm2;
    private final LinearImpl lin;

    public ProteinGNN(long inDim, long hiddenDim, long outClasses) {
        super();
        this.embed = new EmbeddingImpl(30, inDim); // 增加嵌入层
        this.gat = new GATConv(inDim, hiddenDim, 4, 0.2);
        this.sage = new SAGEConv(hiddenDim * 4, hiddenDim);
        this.norm1 = new LayerNorm(hiddenDim, 1e-5, true);
        this.trans = new TransformerConv(hiddenDim, hiddenDim, 2);
        this.norm2 = new LayerNorm(hiddenDim * 2, 1e-5, true);
        this.lin = new LinearImpl(hiddenDim * 2 * 3, outClasses);
        register_module("embedding", embed);
        register_module("gat", gat);
        register_module("sage", sage);
        register_module("norm1", norm1);
        register_module("trans", trans);
        register_module("norm2", norm2);
        register_module("lin", lin);
        // 记得 register_module...
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        x = embed.forward(x.to(kLong()));//.squeeze(1); // 必须转长整型
        if (x.dim() == 3) {
            x = x.squeeze(1);
        }
        x = x.to(kFloat());
        x = elu(gat.forward(x, edge_index));
        x = dropout(x, 0.2, is_training());
//
//        // SAGE -> ReLU
        x = relu(sage.forward(x, edge_index));
        Tensor feat1 = norm1.forward(x);

        Tensor feat2 = relu(trans.forward(feat1, edge_index));
        feat2 = norm2.forward(feat2);
        Tensor combined_feat = cat(new TensorVector(feat1, feat2), -1);

        // 使用我们在上一步实现的 GlobalPooling
        Tensor batch = zeros(new long[]{x.size(0)}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
//        x = GlobalPooling.pool(x, zeros(new long[]{x.size(0)}, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))), "mean");
        Tensor x_mean = GlobalPooling.pool(combined_feat, batch, "mean");
        Tensor x_max = GlobalPooling.pool(combined_feat, batch, "sum"); // 需要你实现 max 模式
        x = cat(new TensorVector(x_mean, x_max), -1);
        return lin.forward(x); // CrossEntropyLoss 内部会处理 Softmax
    }
}


//public class ProteinGNN extends org.bytedeco.pytorch.nn.Module {
//    private EmbeddingImpl embedding;
//    private GATConv gat;
//    private SAGEConv sage;
//    private TransformerConv trans;
//    private LinearImpl lin;
//    private final long hGat = 4; // gat heads
//    private final long hTrans = 2; // transformer heads
//    
//    
//    public ProteinGNN(long inDim, long hiddenDim, long outClasses, long embedding_dim) {
//        super();
//        // 21 代表 20 种氨基酸 + 1 个填充位，128 是嵌入维度
//        this.embedding = new EmbeddingImpl(21, embedding_dim);
//      
//        // 第一层：使用 GAT 捕获局部重要氨基酸
//        this.gat = new GATConv(inDim, hiddenDim, hGat);//, true, 0.2); // 4 heads
//
//        // 第二层：使用 SAGE 聚合邻域平均信息 (hiddenDim * 4 是因为 GAT concat 了 heads)
//        this.sage = new SAGEConv(hiddenDim * hGat, hiddenDim);
//
//        // 第三层：使用 Transformer 建模复杂的相互作用
//        this.trans = new TransformerConv(hiddenDim, hiddenDim, hTrans);
//
//        // 输出层：分类
//        this.lin = new LinearImpl(hiddenDim * hTrans, outClasses);
//        register_module("embedding", embedding);
//        register_module("gat", gat);
//        register_module("sage", sage);
//        register_module("trans", trans);
//        register_module("lin", lin);
//    }
//
//    public Tensor forward(Tensor xs, Tensor edge_index) {
//        // GAT -> ELU -> Dropout
//        Tensor embX = embedding.forward(xs.to(kLong()));
//        var x = gat.forward(embX, edge_index);
//        x = elu(gat.forward(x, edge_index));
//        x = dropout(x, 0.2, is_training());
//
//        // SAGE -> ReLU
//        x = relu(sage.forward(x, edge_index));
//
//        // Transformer -> Global Mean Pool (这里简化为节点表示的均值)
//        x = trans.forward(x, edge_index);
//
//        // 全局聚合：假设预测整个蛋白质的功能
//        x = x.mean(new long[]{0}, false,new ScalarTypeOptional( kFloat()));
//
//        return lin.forward(x.unsqueeze(0)); // [1, outClasses]
//    }
//}