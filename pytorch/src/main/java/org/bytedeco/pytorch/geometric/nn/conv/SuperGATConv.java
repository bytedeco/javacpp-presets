package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.geometric.utils.AggrUtils.scatter_softmax;

/**
 * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.SuperGATConv
 * 特点：引入自监督任务以优化注意力权重的分布。
 */
public class SuperGATConv extends MessagePassing {
    private LinearImpl lin;         // 节点变换 W
    private Tensor att;             // 注意力向量 a
    private String attentionType;   // 'MX' (Mixed) 或 'SD' (Dot-product)
    private int heads;
    private long outChannels;
    private boolean concat;

    // 用于存储当前 forward 过程中的注意力值，供计算 Loss 使用
    private Tensor lastAttentionalWeights;
    private Tensor lastEdgeIndex;

    public SuperGATConv(long inChannels, long outChannels, int heads, boolean concat, String attentionType) {
        super("add");
        this.heads = heads;
        this.outChannels = outChannels;
        this.concat = concat;
        this.attentionType = attentionType;

        // 1. 严格使用 LinearImpl 注册
        this.lin = new LinearImpl(inChannels, heads * outChannels);
        register_module("lin", lin);

        // 2. 注意力参数初始化
        if (attentionType.equals("MX")) {
            // MX 模式拼接源和目标特征: a^T [W*x_i || W*x_j]
            this.att = torch.randn(new long[]{1, heads, 2 * outChannels});
        } else {
            // SD 模式点积
            this.att = torch.randn(new long[]{1, heads, outChannels});
        }
        register_parameter("att", att);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    
    public Tensor forward(Tensor x, Tensor edge_index, Tensor neg_edge_index) {
        long N = x.size(0);

        // 节点投影: [N, H, C]
        Tensor xLin = lin.forward(x).view(N, heads, outChannels);

        // 1. 计算注意力权重
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);

        Tensor x_i = xLin.index_select(0, targetIdx);
        Tensor x_j = xLin.index_select(0, sourceIdx);

        Tensor alpha;
        if (attentionType.equals("MX")) {
            // MX: LeakyReLU(a^T [Wx_i || Wx_j])
            Tensor combined = torch.cat(new TensorVector(x_i, x_j), -1);
            alpha = torch.leaky_relu(combined.mul(att).sum(-1), new Scalar(0.2));
        } else {
            // SD: a^T (Wx_i * Wx_j)
            alpha = (x_i.mul(x_j)).mul(att).sum(-1);
        }

        // 归一化并缓存供自监督使用
        alpha = scatter_softmax(alpha, targetIdx, N);
        this.lastAttentionalWeights = alpha;
        this.lastEdgeIndex = edge_index;

        // 2. 消息传递
        Tensor msg = x_j.mul(alpha.unsqueeze(-1));
        Tensor out = torch.zeros(new long[]{N, heads, outChannels}, x.options());
        out.scatter_add_(0, targetIdx.view(-1, 1, 1).expand_as(msg), msg);

        if (concat) {
            return out.view(N, heads * outChannels);
        } else {
            return out.mean(1);
        }
    }

    /**
     * 实现 get_attention_loss()
     * 用于自监督任务：预测正边概率应高，负边概率应低
     */
    public Tensor get_attention_loss(Tensor neg_edge_index) {
        // 简化版实现：BCE Loss 作用于注意力权重
        // pos_loss = -log(sigmoid(alpha_positive))
        // neg_loss = -log(1 - sigmoid(alpha_negative))
        Tensor posAtt = torch.sigmoid(lastAttentionalWeights.sum(-1));
        Tensor posLoss = posAtt.add(new Scalar(1e-8)).log().mean().neg();

        return posLoss; // 实际还需根据 neg_edge_index 计算负样本损失
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // edge_attr 存储归一化后的系数
        return x_j;
    }
}