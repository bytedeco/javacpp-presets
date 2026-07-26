package org.bytedeco.pytorch.geometric.demo.trainer;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConv;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConvV2;
import org.bytedeco.pytorch.geometric.nn.conv.TransformerConv;
import org.bytedeco.pytorch.geometric.nn.kge.ComplEx;
import org.bytedeco.pytorch.geometric.nn.kge.TransE;

import static org.bytedeco.pytorch.global.torch.*;

public class GridKGETransformer extends org.bytedeco.pytorch.nn.Module {
    // 1. KGE 组件
    private final TransE transE;
    private final ComplEx complEx;

    // 2. GNN 组件
    private final SAGEConvV2 sage;
    private final TransformerConv transConv;
    private final LinearImpl post_lin;
    private final LinearImpl predictor;
    private final long hiddenDim;

    public GridKGETransformer(long numNodes, long numRels, long kgeDim, long historySteps) {
        super();
        this.hiddenDim = kgeDim;

        // 初始化 TransE 和 ComplEx
        this.transE = new TransE(numNodes, numRels, kgeDim, 1);
        this.complEx = new ComplEx(numNodes, numRels, kgeDim);

        // GNN 层：输入维度是历史步长(60) + KGE生成的特征(kgeDim * 2)
        // 这里的 kgeDim * 2 是因为我们融合 TransE 和 ComplEx 的输出
        this.sage = new SAGEConvV2(historySteps + kgeDim * 2, kgeDim, kgeDim);
        this.transConv = new TransformerConv(kgeDim, kgeDim, 4); // 4 heads
        this.post_lin = new LinearImpl(kgeDim * 4, kgeDim * 2);
        this.predictor = new LinearImpl(kgeDim * 2, 1);

        register_module("transE", transE);
        register_module("complEx", complEx);
        register_module("sage", sage);
        register_module("transConv", transConv);
        register_module("post_lin", post_lin);
        register_module("predictor", predictor);
    }

    public Tensor forward(GridKGEData.PowerSnapshot data) {
        // 1. 从 KGE 获取知识图谱表征 (这里取 Embedding 层的权重作为节点特征增强)
        // 注意：不直接使用普通的 Embedding，而是使用 KGE 模型中维护的实体向量
        Tensor transEVec = transE.node_embeddings().weight();
        Tensor complExVec = complEx.node_embeddings().weight(); // 取实部
        Tensor x_norm = data.x.divide(new Scalar(100.0));
        // 2. 拼接原始负荷历史与 KGE 知识特征
        Tensor h = cat(new TensorVector(x_norm, transEVec, complExVec), 1).contiguous();

        // 3. SAGE 聚合空间邻域
        h = relu(sage.forward(h, data.edge_index));

        // 4. TransformerConv 处理动态调度权重
        h = transConv.forward(h, data.edge_index);

        // 5. 最终负荷预测
        h = relu(post_lin.forward(h));
        return predictor.forward(h).squeeze(1);
    }

    // 辅助计算 KGE 损失以训练 KG 空间
//    public Tensor calculateKgeloss(GridKGEData.PowerSnapshot data) {
//        Tensor teLoss = transE.forward(data.headIndices, data.relIndices, data.tailIndices);
//        Tensor ceLoss = complEx.forward(data.headIndices, data.relIndices, data.tailIndices);
//        return teLoss.add(ceLoss).mean();
//    }
    public Tensor calculateKgeloss(GridKGEData.PowerSnapshot data) {
        // 正样本得分
        Tensor pS = transE.forward(data.headIndices, data.relIndices, data.tailIndices)
                .add(complEx.forward(data.headIndices, data.relIndices, data.tailIndices));

        // 负样本 (打乱尾节点)
        Tensor negT = randint(0, data.x.size(0), data.tailIndices.shape(), data.tailIndices.options());
        Tensor nS = transE.forward(data.headIndices, data.relIndices, negT)
                .add(complEx.forward(data.headIndices, data.relIndices, negT));

        // 计算 Margin Loss
        // 只有当正样本得分不比负样本高出 1.0 时，才会产生正值的 Loss
        MarginRankingLossOptions opt = new MarginRankingLossOptions();
        opt.margin().put(1.0);
        return margin_ranking_loss(pS, nS, ones(pS.shape(), pS.options()), opt);
    }

    public Tensor getKgeScore(Tensor h, Tensor r, Tensor t) {
        // 融合 TransE 和 ComplEx 的打分逻辑
        return transE.forward(h, r, t).add(complEx.forward(h, r, t));
    }
}
