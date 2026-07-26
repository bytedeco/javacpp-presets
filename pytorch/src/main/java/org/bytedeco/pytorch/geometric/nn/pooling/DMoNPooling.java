package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

public class DMoNPooling extends Module {
    private LinearImpl mlp;

    public DMoNPooling(long inChannels, long numClusters) {
        this.mlp = new LinearImpl(inChannels, numClusters);
        register_module("mlp", mlp);
    }

    public Tensor[] dmonPool(Tensor x, Tensor adj) {
        // 1. Assignment S
        Tensor s = torch.softmax(mlp.forward(x), 2); // [B, N, K]

        // 2. Pooling
        Tensor xPool = s.transpose(1, 2).matmul(x);
        Tensor adjPool = s.transpose(1, 2).matmul(adj).matmul(s);

        // 3. DMoN Loss (Modularity)
        // B = A - (d d^T) / 2m
        Tensor deg = adj.sum(new long[]{2}, false, new ScalarTypeOptional()).unsqueeze(2); // [B, N, 1]
        Tensor degT = deg.transpose(1, 2); // [B, 1, N]

        Tensor m2 = deg.sum(); // 2 * num_edges (Total volume)

        // Modularity Matrix B: [B, N, N]
        Tensor B_mod = adj.sub(deg.matmul(degT).div(m2));

        // Spectral Modularity = Tr(S^T B S)
        Tensor sT_B_s = s.transpose(1, 2).matmul(B_mod).matmul(s);
        Tensor modularity = torch.einsum("bkk->b", new TensorVector(sT_B_s)).mean();

        // DMoN Loss = - Modularity + Collapse Regularization
        // Collapse Reg: || S^T S ||_F / N_nodes  (Force clusters to be roughly equal size)
        long N = x.size(1);
        long K = s.size(2);

        // Original paper regularization:
        // sqrt(K)/N * || S^T S || - 1
        Tensor sTs = s.transpose(1, 2).matmul(s);
        Tensor collapseLoss = sTs.norm().mul(new Scalar(Math.sqrt(K))).div(new Scalar(N)).sub(new Scalar(1)).mean();

        return new Tensor[]{xPool, adjPool, modularity.neg(), collapseLoss, s};
    }
}