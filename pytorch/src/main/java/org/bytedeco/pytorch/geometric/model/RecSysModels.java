package org.bytedeco.pytorch.geometric.model;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GATConv;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConv;
//import org.gnn.framework.layers.org.bytedeco.pytorch.geometric.nn.conv.SAGEConv;
//import org.gnn.framework.layers.org.bytedeco.pytorch.geometric.nn.conv.GATConv;

import static org.bytedeco.pytorch.global.torch.*;

public class RecSysModels {

    /**
     * ===================================================================
     * 模型 1: 电商商品召回模型 (基于 GraphSAGE)
     * 目标: 生成 User 和 Item 的 Embedding，计算向量相似度
     * ===================================================================
     */
    public static class EcommerceRecallModel extends Module {
        private SAGEConv conv1;
        private SAGEConv conv2;

        public EcommerceRecallModel(long inChannels, long hiddenChannels, long outChannels) {
            super();
            // 第一层: 聚合邻居信息
            this.conv1 = new SAGEConv(inChannels, hiddenChannels);
            // 第二层: 生成最终 Embedding
            this.conv2 = new SAGEConv(hiddenChannels, outChannels);

            register_module("conv1", conv1);
            register_module("conv2", conv2);
        }

        // 返回所有节点的 Embedding
        public Tensor forward(Tensor x, Tensor edge_index) {
            Tensor h = conv1.forward(x, edge_index);
            h = torch.relu(h);
            h = torch.dropout(h, 0.5, is_training());

            h = conv2.forward(h, edge_index);
            // 召回任务通常会对 Embedding 做归一化，使得点积等于余弦相似度
            return torch.normalize(h);
        }

        // 辅助方法：计算成对得分 (Dot Product)
        // srcEmbeds: [Batch, Dim], dstEmbeds: [Batch, Dim] -> [Batch]
        public Tensor predictSimilarity(Tensor srcEmbeds, Tensor dstEmbeds) {
            return srcEmbeds.mul(dstEmbeds).sum(new long[]{1}, false,new ScalarTypeOptional(ScalarType.Float));
        }
    }

    /**
     * ===================================================================
     * 模型 2: 社交网络相亲排序模型 (基于 org.bytedeco.pytorch.geometric.nn.model.GAT + MLP)
     * 目标: 给定两个 User，预测匹配分值 (0-1)
     * ===================================================================
     */
    public static class SocialRankingModel extends Module {
        private GATConv conv1;
        private GATConv conv2;
        private SequentialImpl predictor; // 链路评分器 (MLP)
        private double negativeSlope = 0.2;
        public SocialRankingModel(long inChannels, long hiddenChannels, long outChannels, long heads) {
            super();

            // GNN 部分: 提取具有社交属性的节点特征
            // Layer 1: Concat heads
            this.conv1 = new GATConv(inChannels, hiddenChannels, heads, negativeSlope);
            // Layer 2: Mean heads (得到最终 Node Embedding)
            this.conv2 = new GATConv(hiddenChannels * heads, outChannels, 1, negativeSlope);

            // Predictor 部分: 接收两个用户的 Embedding，输出匹配分
            // 输入维度 = UserA_Dim + UserB_Dim = outChannels * 2
            this.predictor = new SequentialImpl();
            predictor.push_back(new LinearImpl(outChannels * 2, hiddenChannels));
            predictor.push_back(new ReLUImpl());
            predictor.push_back(new LinearImpl(hiddenChannels, 1)); // 输出一个分数

            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("predictor", predictor);
        }

        // 生成节点特征
        public Tensor getNodeEmbeddings(Tensor x, Tensor edge_index) {
            Tensor h = conv1.forward(x, edge_index);
            h = torch.elu(h);
            h = torch.dropout(h, 0.6, is_training());
            h = conv2.forward(h, edge_index);
            return h;
        }

        // 完整前向传播：输入图结构 + 需要打分的边(src, dst)
        public Tensor forward(Tensor x, Tensor edge_index, Tensor srcNodeIdx, Tensor dstNodeIdx) {
            // 1. 获取全图节点的 Embedding
            Tensor z = getNodeEmbeddings(x, edge_index);

            // 2. 提取需要打分的成对节点特征
            Tensor zSrc = z.index_select(0, srcNodeIdx);
            Tensor zDst = z.index_select(0, dstNodeIdx);

            // 3. 特征拼接 (Concat)
            // [Batch, Dim] cat [Batch, Dim] -> [Batch, 2*Dim]
            Tensor catFeat = torch.cat(new TensorVector(zSrc, zDst), 1);

            // 4. MLP 评分
            return predictor.forward(catFeat);
        }
    }
}