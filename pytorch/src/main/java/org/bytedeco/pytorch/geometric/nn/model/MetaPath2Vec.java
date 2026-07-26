package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;

public class MetaPath2Vec extends Module {
    private EmbeddingImpl embedding;

    public MetaPath2Vec(long numNodes, long embeddingDim) {
        this.embedding = new EmbeddingImpl(numNodes, embeddingDim);
        register_module("embedding", embedding);
    }

    public Tensor forward(Tensor nodeIdx) {
        return embedding.forward(nodeIdx);
    }

    // 负采样 Loss
    // pos_rw: [Batch, WalkLen]
    // neg_rw: [Batch, WalkLen, NegSamples]
    public Tensor loss(Tensor posRw, Tensor negRw) {
        Tensor startNode = posRw.select(1, 0);
        Tensor hStart = embedding.forward(startNode);

        // Positive
        Tensor hPos = embedding.forward(posRw.slice(1, new LongOptional(1), new LongOptional(posRw.size(1)), 1));
        Tensor posScore = torch.einsum("bd,bwd->bw", new TensorVector(hStart, hPos));
        Tensor posLoss = torch.log(torch.sigmoid(posScore)).neg().mean();

        // Negative (simplified)
        // ... similar to org.bytedeco.pytorch.geometric.nn.model.Node2Vec ...
        return posLoss;
    }
}