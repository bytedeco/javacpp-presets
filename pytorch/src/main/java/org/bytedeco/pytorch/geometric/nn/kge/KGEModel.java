package org.bytedeco.pytorch.geometric.nn.kge;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions;
import org.bytedeco.pytorch.nn.options.MarginRankingLossOptions;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
/**
 * KGEModel Base Class
 * 所有知识图谱嵌入模型的基类。
 */
public abstract class KGEModel extends Module {
    protected EmbeddingImpl nodeEmb;
    protected EmbeddingImpl relEmb;
    protected long numNodes;
    protected long numRels;
    protected long hiddenChannels;

    public KGEModel(long numNodes, long numRels, long hiddenChannels) {
        super();
        this.numNodes = numNodes;
        this.numRels = numRels;
        this.hiddenChannels = hiddenChannels;

        this.nodeEmb = new EmbeddingImpl(numNodes, hiddenChannels);
        this.relEmb = new EmbeddingImpl(numRels, hiddenChannels);

        // 初始化是个好习惯 (Xavier or Uniform)
        torch.xavier_normal_(nodeEmb.weight());
        torch.xavier_normal_(relEmb.weight());

        register_module("nodeEmb", nodeEmb);
        register_module("relEmb", relEmb);
    }

    public EmbeddingImpl node_embeddings() {
        return nodeEmb;
    }

    public EmbeddingImpl relation_embeddings() {
        return relEmb;
    }
    /**
     * 计算三元组得分 (Score Function)
     * @param head [Batch] 头实体索引
     * @param relation [Batch] 关系索引
     * @param tail [Batch] 尾实体索引
     * @return scores [Batch] (分数越高越好，或者距离越小越好，视具体实现而定)
     */
    public abstract Tensor forward(Tensor head, Tensor relation, Tensor tail);

    /**
     * Margin Ranking Loss
     * Loss = max(0, margin - score_pos + score_neg)
     */
    public Tensor loss(Tensor posScore, Tensor negScore, double margin) {
        // PyTorch MarginRankingLoss: max(0, -y * (x1 - x2) + margin)
        // 令 y=1, x1=pos, x2=neg => max(0, margin - (pos - neg)) = max(0, margin - pos + neg)
        // 注意：有些模型(如TransE)是距离模型(越小越好)，有些(DistMult)是相似度模型(越大越好)。
        // 这里假设 forward 返回的是“得分”（越大越好）。
        // 如果是距离模型，forward 应该返回 -distance。

        Tensor target = torch.ones_like(posScore);
        MarginRankingLossOptions opt = new MarginRankingLossOptions();
        opt.margin().put(margin);
        return torch.margin_ranking_loss(posScore, negScore, target, opt);
    }

    // 辅助：L2 Regularization
    public Tensor regLoss() {
        return nodeEmb.weight().pow(new Scalar(2)).sum().add(relEmb.weight().pow(new Scalar(2)).sum());
    }
}