package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;

public class DeepGraphInfomax extends Module {
    private Module encoder;      // GNN
    private Module summary;      // Readout function (e.g., Mean/Sigmoid)
    private Module corruption;   // Corruption function (shuffles features)
    private LinearImpl discriminator; // Bilinear or simple MLP

    public DeepGraphInfomax(Module encoder, long hiddenDim) {
        this.encoder = encoder;
        this.discriminator = new LinearImpl(hiddenDim, hiddenDim); // Simplified bilinear
        register_module("encoder", encoder);
        register_module("discriminator", discriminator);
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. Positive Samples
        Tensor posZ = ((GCNConv)encoder).forward(x, edge_index); // 假设是 GCN

        // 2. Summary (Global Context) s = Sigmoid(Mean(posZ))
        Tensor summaryVec = posZ.mean(new long[]{0}, true,new ScalarTypeOptional()).sigmoid(); // [1, F]

        // 3. Negative Samples (Corruption: Shuffle X)
        Tensor negX = x.index_select(0, torch.randperm(x.size(0)));
        Tensor negZ = ((GCNConv)encoder).forward(negX, edge_index);

        // 4. Discriminator Scores
        // Score = Z @ W @ s.T
        // 这里简化为 Bilinear: (Z @ W) * s
        Tensor posScore = discriminator.forward(posZ).matmul(summaryVec.t());
        Tensor negScore = discriminator.forward(negZ).matmul(summaryVec.t());

        return torch.cat(new TensorVector(posScore, negScore), 0);
    }

    // External BCE Loss needed: Label 1 for pos, 0 for neg
}
