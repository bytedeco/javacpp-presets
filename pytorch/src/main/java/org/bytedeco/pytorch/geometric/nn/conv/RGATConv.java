package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;

/**
 * 修复后的 RGATConv (关系图注意力网络)
 * 解决了关系掩码与多头特征维度不匹配的问题。
 */
public class RGATConv extends MessagePassing {
    private long inChannels, outChannels;
    private int numRelations, heads;
    private boolean concat;
    private long dK;

    private LinearImpl lin; // 节点投影
    private Tensor relAtt;  // 关系特定注意力参数 [numRelations, heads, 2 * dK]

    public RGATConv(long inChannels, long outChannels, int numRelations, int heads, boolean concat) {
        super("add");
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.numRelations = numRelations;
        this.heads = heads;
        this.concat = concat;
        this.dK = outChannels / heads;

//        this.lin = new LinearImpl(inChannels, outChannels);
        this.lin = new LinearImpl(inChannels, heads * outChannels);
        register_module("lin", lin);

        // 关系注意力向量
//        this.relAtt = torch.randn(new long[]{numRelations, heads, 2 * dK});
        this.relAtt = torch.randn(new long[]{numRelations, heads, 2 * outChannels});
        register_parameter("rel_att", relAtt);
    }

    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_type) {
        long N = x.size(0);
        long E = edge_index.size(1);

        // 1. 线性投影并重塑为多头 [N, H, dK]
//        Tensor xLin = lin.forward(x).view(N, heads, dK);
        Tensor xLin = lin.forward(x).view(N, heads, outChannels);
        // 2. 初始化每条边的消息 [E, H, dK]
//        Tensor edgeMessages = torch.zeros(new long[]{E, heads, dK}, x.options());

        // 4. 初始化每条边的消息 [E, heads, outChannels]
        Tensor edgeMessages = torch.zeros(new long[]{E, heads, outChannels}, x.options());
        for (int r = 0; r < numRelations; r++) {
            // 获取当前关系的边掩码 [E]
            Tensor mask = edge_type.eq(new Scalar(r));

            if (mask.any().item_bool()) {
                Tensor relEdgeIndex = edge_index.masked_select(mask.unsqueeze(0).expand(new long[]{2, E})).view(2, -1);
                Tensor row = relEdgeIndex.select(0, 0);
                Tensor col = relEdgeIndex.select(0, 1);

                // 提取源和目标特征 [Er, H, dK]
                Tensor x_j = xLin.index_select(0, row);
                Tensor x_i = xLin.index_select(0, col);

                // 计算注意力 (此处简化为求和后变换)
                Tensor alpha = torch.cat(new TensorVector(x_i, x_j), -1); // [Er, H, 2*dK]
                Tensor att = relAtt.select(0, r); // [H, 2*dK]

                // 计算得分并应用 Softmax
                Tensor score = (alpha.mul(att)).sum(-1); // [Er, H]
                // 模拟 softmax 并对消息加权
                Tensor relMsg = x_j.mul(torch.sigmoid(score).unsqueeze(-1));

                // 3. 关键修复：将 [Er, H, dK] 的结果散布回全边张量
                // 构造匹配 [E, H, dK] 的掩码
                Tensor expandedMask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(edgeMessages);
                edgeMessages.masked_scatter_(expandedMask, relMsg);
            }
        }

        // 4. 聚合消息并处理输出
        Tensor out = aggregate(edgeMessages, edge_index.select(0, 1), N); // [N, H, dK]

        if (concat) {
            return out.view(N, heads* outChannels);
        } else {
            return out.mean(1);
        }
    }

        @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }

        @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 由于上面使用了自定义的 propagate_gated，这里的基类实现作为备用签名
        return x_j;
    }
}
/**
 * 实现 torch_geometric.nn.conv.RGATConv
 * 关系图注意力算子，支持多关系下的多头注意力机制。
 */
//public class RGATConv extends MessagePassing {
//    private long inChannels, outChannels;
//    private int numRelations, heads;
//    private boolean concat;
//
//    // 为每个关系准备独立的投影矩阵
//    private LinearImpl[] linQueries;
//    private LinearImpl[] linKeys;
//    private LinearImpl[] linValues;
//    private Tensor[] atts; // 关系特定的注意力向量
//
//    public RGATConv(long inChannels, long outChannels, int numRelations, int heads, boolean concat) {
//        super("add");
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.numRelations = numRelations;
//        this.heads = heads;
//        this.concat = concat;
//
//        this.linQueries = new LinearImpl[numRelations];
//        this.linKeys = new LinearImpl[numRelations];
//        this.linValues = new LinearImpl[numRelations];
//        this.atts = new Tensor[numRelations];
//
//        for (int r = 0; r < numRelations; r++) {
//            // Query, Key, Value 投影
//            linQueries[r] = new LinearImpl(inChannels, heads * outChannels);
//            linKeys[r] = new LinearImpl(inChannels, heads * outChannels);
//            linValues[r] = new LinearImpl(inChannels, heads * outChannels);
//
//            // 注意力向量 a_r [1, heads, outChannels]
//            atts[r] = torch.randn(new long[]{1, heads, outChannels});
//            torch.xavier_uniform_(atts[r]);
//
//            register_module("lin_q_" + r, linQueries[r]);
//            register_module("lin_k_" + r, linKeys[r]);
//            register_module("lin_v_" + r, linValues[r]);
//            register_parameter("att_" + r, atts[r]);
//        }
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, (Tensor)null);
//    }
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_type) {
//        long N = x.size(0);
//        long E = edge_index.size(1);
//
//        // 1. 投影节点特征到多头空间 [N, H, D]
//        Tensor x_lin = lin.forward(x).view(N, heads, outChannels / heads);
//
//        // 2. 初始化输出和注意力存储
//        Tensor out = torch.zeros(new long[]{E, heads, outChannels / heads}, x.options());
//
//        // 3. 核心修复：按关系处理
//        for (int r = 0; r < numRelations; r++) {
//            // 获取当前关系的边掩码 [E]
//            Tensor mask = edge_type.eq(new Scalar(r));
//
//            // 检查当前关系是否有边
//            if (mask.any().item_bool()) {
//                // 提取当前关系的源/目标索引
//                Tensor row = edge_index.select(0, 0).masked_select(mask);
//                Tensor col = edge_index.select(0, 1).masked_select(mask);
//
//                // 执行关系特定的注意力计算 (逻辑简化)
//                Tensor rel_out = compute_relation_attention(x_lin, row, col, r);
//
//                // 关键修复点：将 [E] 的掩码扩展为 [E, Heads, 1] 以便匹配 [E, Heads, D]
//                // 或者展平后进行散布
//                Tensor expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(out);
//                out.masked_scatter_(expanded_mask, rel_out);
//            }
//        }
//
//        // 4. 聚合与多头拼接
//        Tensor res = aggregate(out, edge_index.select(0, 1), N);
//        return concat ? res.view(N, outChannels) : res.mean(1);
//    }
//    public Tensor forward2(Tensor x, Tensor edge_index, Tensor edge_type) {
//        long N = x.size(0);
//        long E = edge_index.size(1);
//
//        // 1. 初始化全图 Logits [E, heads] 和 Value [E, heads, outChannels]
//        Tensor allLogits = torch.full(new long[]{E, heads}, new Scalar(-1e9), x.options());
//        Tensor allValues = torch.zeros(new long[]{E, heads, outChannels}, x.options());
//
//        // 2. 按关系迭代计算 (Across-relation 逻辑)
//        for (int r = 0; r < numRelations; r++) {
//            Tensor mask = edge_type.eq(new Scalar(r));
//            if (!mask.any().item_bool()) continue;
//
//            // 提取该关系的局部索引
//            Tensor edge_r = edge_index.masked_select(mask.unsqueeze(0).expand(new long[]{2, E})).view(2, -1);
//            Tensor src_r = edge_r.select(0, 0);
//            Tensor dst_r = edge_r.select(0, 1);
//
//            // 投影: [N, H, C]
//            Tensor Q = linQueries[r].forward(x).view(N, heads, outChannels);
//            Tensor K = linKeys[r].forward(x).view(N, heads, outChannels);
//            Tensor V = linValues[r].forward(x).view(N, heads, outChannels);
//
//            // 计算注意力 Logits: LeakyReLU(a^T * (Q_i + K_j))
//            Tensor Qi = Q.index_select(0, dst_r);
//            Tensor Kj = K.index_select(0, src_r);
//            Tensor logits_r = torch.leaky_relu(Qi.add(Kj), new Scalar(0.2)).mul(atts[r]).sum(-1);
//
//            // 填回全局张量
//            allLogits.masked_scatter_(mask, logits_r);
//            allValues.masked_scatter_(mask.view(-1, 1, 1).expand_as(allValues.index_select(0, torch.arange(new Scalar(E)).masked_select(mask))),
//                    V.index_select(0, src_r));
//        }
//
//        // 3. Across-relation Softmax 归一化
//        Tensor targetIdx = edge_index.select(0, 1);
//        Tensor alpha = scatter_softmax(allLogits, targetIdx, N);
//
//        // 4. 加权聚合
//        Tensor msg = allValues.mul(alpha.unsqueeze(-1));
//        Tensor out = torch.zeros(new long[]{N, heads, outChannels}, x.options());
//        out.scatter_add_(0, targetIdx.view(-1, 1, 1).expand_as(msg), msg);
//
//        // 5. 合并多头
//        if (concat) {
//            return out.view(N, heads * outChannels);
//        } else {
//            return out.mean(1);
//        }
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 由于上面使用了自定义的 propagate_gated，这里的基类实现作为备用签名
//        return x_j;
//    }
//    private Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
//        Tensor maxVal = Scatter.scatter(src, index, dimSize, "max");
//        Tensor out = src.sub(maxVal.index_select(0, index)).exp();
//        Tensor sum = Scatter.scatter(out, index, dimSize, "add");
//        return out.div(sum.index_select(0, index).add(new Scalar(1e-16)));
//    }
//}