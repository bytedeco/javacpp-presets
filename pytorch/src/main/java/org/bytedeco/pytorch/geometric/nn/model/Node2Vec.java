package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;

public class Node2Vec extends Module {
    private EmbeddingImpl embedding;
    private long numNodes;
    private long embeddingDim;

    // Context window size, walk length 等参数主要用于采样过程，模型只负责 Embedding

    public Node2Vec(long numNodes, long embeddingDim) {
        this.numNodes = numNodes;
        this.embeddingDim = embeddingDim;
        this.embedding = new EmbeddingImpl(numNodes, embeddingDim);
        // sparse=true for SparseAdam if needed
        register_module("embedding", embedding);
    }

    public Tensor forward(Tensor batch) {
        return embedding.forward(batch);
    }

    /**
     * Loss function (Negative Sampling)
     * @param posRw [Batch, Window] 正样本游走序列
     * @param negRw [Batch, Window, NegSamples] 负样本
     */
    public Tensor loss(Tensor posRw, Tensor negRw) {
        // Start node is posRw[:, 0]
        Tensor startNode = posRw.select(1, 0);
        Tensor hStart = embedding.forward(startNode); // [Batch, Dim]

        // Positive Loss: -log(sigmoid(h_start * h_pos))
        Tensor hPos = embedding.forward(posRw.slice(1, new LongOptional(1),new LongOptional( posRw.size(1)), 1)); // [Batch, Win-1, Dim]
        Tensor posScore = torch.einsum("bd,bwd->bw", new TensorVector(hStart, hPos));
        Tensor posLoss = torch.log(torch.sigmoid(posScore)).neg().mean();

        // Negative Loss
        // ... 类似逻辑 ...

        return posLoss; // + negLoss
    }
}