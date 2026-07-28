package org.bytedeco.pytorch.geometric.nn.kge;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.MarginRankingLossOptions;

/**
 * Base class for Knowledge Graph Embedding models.
 *
 * <p>Holds entity/relation embeddings and defines the triple scoring API.
 * Subclasses implement {@link #forward(Tensor, Tensor, Tensor)} returning a
 * <b>similarity score</b> (higher = more plausible). Distance-based models
 * (TransE, RotatE) should return {@code -distance}.
 */
public abstract class KGEModel extends Module {

    protected final EmbeddingImpl nodeEmb;
    protected final EmbeddingImpl relEmb;
    protected final long numNodes;
    protected final long numRels;
    protected final long hiddenChannels;

    public KGEModel(long numNodes, long numRels, long hiddenChannels) {
        super();
        if (numNodes <= 0 || numRels <= 0 || hiddenChannels <= 0) {
            throw new IllegalArgumentException("numNodes/numRels/hiddenChannels must be > 0");
        }
        this.numNodes = numNodes;
        this.numRels = numRels;
        this.hiddenChannels = hiddenChannels;

        this.nodeEmb = register_module("nodeEmb", new EmbeddingImpl(numNodes, hiddenChannels));
        this.relEmb = register_module("relEmb", new EmbeddingImpl(numRels, hiddenChannels));
        torch.xavier_uniform_(nodeEmb.weight());
        torch.xavier_uniform_(relEmb.weight());
    }

    /**
     * Triple score (higher = better).
     *
     * @param head     [B] long entity ids
     * @param relation [B] long relation ids
     * @param tail     [B] long entity ids
     * @return [B] scores
     */
    public abstract Tensor forward(Tensor head, Tensor relation, Tensor tail);

    /**
     * Margin ranking loss assuming higher scores are better:
     * {@code max(0, margin − s⁺ + s⁻)}.
     */
    public Tensor loss(Tensor posScore, Tensor negScore, double margin) {
        if (posScore == null || negScore == null) {
            throw new NullPointerException("posScore/negScore must not be null");
        }
        Tensor target = torch.ones_like(posScore); // prefer pos > neg
        MarginRankingLossOptions opt = new MarginRankingLossOptions();
        opt.margin().put(margin);
        return torch.margin_ranking_loss(posScore, negScore, target, opt);
    }

    /** L2 regularization on all embedding weights. */
    public Tensor regLoss() {
        return nodeEmb.weight().pow(new Scalar(2)).mean()
                .add(relEmb.weight().pow(new Scalar(2)).mean());
    }

    public EmbeddingImpl node_embeddings() {
        return nodeEmb;
    }

    public EmbeddingImpl relation_embeddings() {
        return relEmb;
    }

    public long getNumNodes() {
        return numNodes;
    }

    public long getNumRels() {
        return numRels;
    }

    public long getHiddenChannels() {
        return hiddenChannels;
    }
}
