package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.HEATConv
 * 异构边增强型图注意力算子，支持节点/边类型编码及边特征融合。
 */
public class HEATConv extends MessagePassing {
    private int numNodeTypes, numEdgeTypes, heads;
    private long outChannels;
    private boolean concat;

    // 1. 节点类型投影: 每一类节点拥有独立的 LinearImpl
    private LinearImpl[] nodeLins;

    // 2. 边增强组件
    private EmbeddingImpl edgeTypeEmb;   // 边类型嵌入
    private LinearImpl edgeAttrLin;  // 边属性投影

    // 3. 注意力与消息层
    private LinearImpl linAtt;       // 注意力 Score 计算 [d_h * 3, 1]
    private LinearImpl linRoot;      // 自连接权重 (Root weight)

    private Tensor bias;

    public HEATConv(long inChannels, long outChannels, int numNodeTypes, int numEdgeTypes,
                    int edgeTypeEmbDim, int edgeDim, int edgeAttrEmbDim, int heads, boolean concat) {
        super("add");
        this.outChannels = outChannels;
        this.numNodeTypes = numNodeTypes;
        this.numEdgeTypes = numEdgeTypes;
        this.heads = heads;
        this.concat = concat;

        // 严格使用 LinearImpl 注册节点类型映射
        this.nodeLins = new LinearImpl[numNodeTypes];
        for (int i = 0; i < numNodeTypes; i++) {
            nodeLins[i] = new LinearImpl(inChannels, heads * outChannels);
            register_module("lin_node_" + i, nodeLins[i]);
        }

        // 边特征处理
        this.edgeTypeEmb = new EmbeddingImpl(numEdgeTypes, edgeTypeEmbDim);
        this.edgeAttrLin = new LinearImpl(edgeDim, edgeAttrEmbDim);
        register_module("edge_type_emb", edgeTypeEmb);
        register_module("edge_attr_lin", edgeAttrLin);

        // 注意力机制: 融合节点特征、边类型特征和边属性特征
        // 输入维度 = (heads * outChannels) * 2 + edgeTypeEmbDim + edgeAttrEmbDim
        long attInDim = (heads * outChannels) * 2 + edgeTypeEmbDim + edgeAttrEmbDim;
        this.linAtt = new LinearImpl(attInDim, heads);
        register_module("lin_att", linAtt);

        if (concat) {
            this.linRoot = new LinearImpl(inChannels, heads * outChannels);
        } else {
            this.linRoot = new LinearImpl(inChannels, outChannels);
        }
        register_module("lin_root", linRoot);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 假设：
        // x_j (源节点投影后): [E, heads * outChannels] -> [E, 64]
        // x_i (目标节点投影后): [E, heads * outChannels] -> [E, 64]
        // edge_attr (预处理后的边特征): [E, edgeTypeEmbDim + edgeAttrEmbDim] -> [E, 16]

        // 1. 构造注意力输入：拼接源、目标节点特征以及边特征
        // [E, 64 + 64 + 16] = [E, 144]
        Tensor attInput = torch.cat(new TensorVector(x_i, x_j, edge_attr), -1);

        // 2. 计算注意力得分 alpha: [E, heads] -> [E, 2]
        Tensor alpha = linAtt.forward(attInput);

        // 3. 激活与归一化 (针对每个 head 独立做 softmax 可选，通常在基类 aggregate 前做)
        alpha = torch.leaky_relu(alpha, new Scalar(0.2));
        // 这里我们可以先不做 softmax，留在 aggregate 阶段配合 Softmax 算子处理，
        // 或者在这里做简易缩放

        // 4. 多头广播机制 (关键修正：解决 64 vs 8 匹配问题)
        // a) 将源节点特征拆分为多头形式: [E, heads, outChannels] -> [15, 2, 32]
        Tensor msg = x_j.view(new long[]{-1, heads, outChannels});

        // b) 将注意力得分扩展维度以便广播: [E, heads] -> [E, heads, 1] -> [15, 2, 1]
        Tensor alphaReshaped = alpha.unsqueeze(-1);

        // 5. 逐头加权
        // [15, 2, 32] * [15, 2, 1] = [15, 2, 32]
        Tensor out = msg.mul(alphaReshaped);

        // 6. 展平回 2D 形状返回给 aggregate
        // [15, 64]
        return out.view(new long[]{-1, heads * outChannels});
    }
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 1. 准备边特征 (假设 eType 和 eAttr 已经由 forward 预处理好存入成员变量)
//        // 或者在 forward 中拼接好传给 propagate 的 edge_attr 参数
//
//        // 2. 计算注意力得分 alpha: [E, heads]
//        // 假设 attInput 是拼接后的 [x_i, x_j, eType, eAttr]
//        Tensor attInput = torch.cat(new TensorVector(x_i, x_j, edge_attr), -1);
//        Tensor alpha = linAtt.forward(attInput);
//        alpha = torch.softmax(alpha, -1); // 在 heads 维度做归一化
//
//        // 3. 维度重塑与广播 (关键修正)
//        // x_j 形状是 [E, heads * outChannels] -> [E, heads, outChannels]
//        Tensor msg = x_j.view(new long[]{-1, heads, outChannels});
//
//        // alpha 形状是 [E, heads] -> [E, heads, 1]
//        Tensor alphaReshaped = alpha.unsqueeze(-1);
//
//        // 4. 执行相乘并展平
//        // [E, heads, outChannels] * [E, heads, 1] -> [E, heads, outChannels]
//        Tensor out = msg.mul(alphaReshaped);
//
//        // 展平回 [E, 64] 以便后续聚合
//        return out.view(new long[]{-1, heads * outChannels});
//    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    
    public Tensor forward(Tensor x, Tensor edge_index, Tensor node_type, Tensor edge_type, Tensor edge_attr) {
        long N = x.size(0);

        // --- 1. 节点类型投影 (Heterogeneous Node Projection) ---
        // 目标：根据节点所属类型，应用对应的 nodeLins[i]
        Tensor xTrans = torch.zeros(new long[]{N, heads * outChannels}, x.options());

        for (int i = 0; i < numNodeTypes; i++) {
            // 找出所有类型为 i 的节点索引
            Tensor mask = node_type.eq(new Scalar(i));

            // 只有当该类型节点存在时才处理，避免空张量操作报错
            if (mask.any().item_bool()) {
                // 提取该类型的特征：[num_type_i, inChannels]
                Tensor typeNodes = x.index_select(0, mask.nonzero().squeeze());

                // 应用对应的线性层：[num_type_i, heads * outChannels]
                Tensor projected = nodeLins[i].forward(typeNodes);

                // 将结果写回对应的位置
                xTrans.masked_scatter_(mask.unsqueeze(-1), projected);
            }
        }

        // --- 2. 边特征准备 (Edge Feature Preparation) ---
        // 边类型 Embedding: [E] -> [E, edgeTypeEmbDim]
        Tensor eType = edgeTypeEmb.forward(edge_type);

        // 边属性线性投影: [E, edgeDim] -> [E, edgeAttrEmbDim]
        Tensor eAttr = edgeAttrLin.forward(edge_attr);

        // 拼接成统一的边属性，传给 message 函数: [E, eTypeDim + eAttrDim]
        Tensor combinedEdgeAttr = torch.cat(new TensorVector(eType, eAttr), -1);

        // --- 3. 消息传递 (Message Passing) ---
        // 调用 propagate，内部会分发 xTrans (作为 x_i, x_j) 和 combinedEdgeAttr (作为 edge_attr)
        Tensor out = propagate(edge_index, xTrans, combinedEdgeAttr);

        // --- 4. 残差连接与自投影 (Root / Self-connection) ---
        // linRoot 将原始输入 x 映射到输出维度
        Tensor res = linRoot.forward(x);

        // 如果 concat 为 true，则维度对齐 heads * outChannels
        // 最终输出: 聚合信息 + 残差信息
        return out.add(res);
    }
    public Tensor forward2(Tensor x, Tensor edge_index, Tensor node_type, Tensor edge_type, Tensor edge_attr) {
        long N = x.size(0);

        // 1. 根据节点类型进行投影
        Tensor xTrans = torch.zeros(new long[]{N, heads * outChannels}, x.options());
        for (int i = 0; i < numNodeTypes; i++) {
            Tensor mask = node_type.eq(new Scalar(i));
            if (mask.any().item_bool()) {
                xTrans.masked_scatter_(mask.unsqueeze(-1), nodeLins[i].forward(x.masked_select(mask.unsqueeze(-1)).view(-1, x.size(1))));
            }
        }

        // 2. 边特征准备
        Tensor eType = edgeTypeEmb.forward(edge_type);
        Tensor eAttr = edgeAttrLin.forward(edge_attr);

        // 3. 消息传递与注意力计算
        Tensor out = propagate(edge_index, xTrans, eType, eAttr);

        // 4. 合并自连接 (Root Weight)
        Tensor rootOut = linRoot.forward(x);
        return out.add(rootOut);
    }

//    @Override
    public Tensor message2(Tensor x_j, Tensor x_i, Tensor eType, Tensor eAttr, long numNodes) {
        // 拼接节点特征与边增强特征计算注意力
        Tensor combined = torch.cat(new TensorVector(x_i, x_j, eType, eAttr), -1);
        Tensor alpha = linAtt.forward(combined);
        alpha = torch.leaky_relu(alpha, new Scalar(0.2));

        // 实际应用中需在此处进行 scatter_softmax 归一化
        return x_j.mul(alpha.unsqueeze(-1));
    }
}