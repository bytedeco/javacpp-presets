package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.*;

public class DenseDiffPool {

    /**
     * @param x Node features [B, N, F]
     * @param adj Adjacency [B, N, N]
     * @param s Assignment matrix [B, N, K]
     * @return Tensor[] {x_pool, adj_pool, link_loss, ent_loss}
     */
    public static Tensor[] dense_diff_pool(Tensor x, Tensor adj, Tensor s) {
        // 1. Coarsen Features: X' = S^T @ X
        // [B, K, N] @ [B, N, F] -> [B, K, F]
        Tensor xPool = s.transpose(1, 2).matmul(x);

        // 2. Coarsen Adj: A' = S^T @ A @ S
        // [B, K, N] @ [B, N, N] -> [B, K, N] @ [B, N, K] -> [B, K, K]
        Tensor adjPool = s.transpose(1, 2).matmul(adj).matmul(s);

        // 3. Auxiliary Losses
        // Link Pred Loss: || A - S @ S^T ||_F
        // S @ S^T approximates A
        Tensor sSt = s.matmul(s.transpose(1, 2));
        Tensor linkLoss = adj.sub(sSt).norm(new Scalar(2)).pow(new Scalar(2)).mean();

        // Entropy Loss: - sum( S * log(S) )
        Tensor entLoss = s.mul(s.add(new Scalar(1e-6)).log()).neg().mean();

        return new Tensor[]{xPool, adjPool, linkLoss, entLoss};
    }
}
