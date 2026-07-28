package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;
public class DenseMinCutPool extends Module {
    private LinearImpl mlp;

    public DenseMinCutPool(long inChannels, long numClusters) {
        this.mlp = new LinearImpl(inChannels, numClusters);
        register_module("mlp", mlp);
    }

    /**
     * @return {x_pool, adj_pool, cut_loss, ortho_loss, s}
     */
    public Tensor[] minCutPool(Tensor x, Tensor adj) {
        // 1. Compute Assignment S = Softmax(MLP(X))
        Tensor s = torch.softmax(mlp.forward(x), 2); // [B, N, K]

        // 2. Pooling (Same as DiffPool)
        Tensor xPool = s.transpose(1, 2).matmul(x);
        Tensor adjPool = s.transpose(1, 2).matmul(adj).matmul(s);

        // 3. MinCut Loss
        // Cut Loss = - Tr(S^T A S) / Tr(S^T D S) (Relaxed Normalized Cut)
        // Numerator: Tr(A_pool)
        // Denominator: D_pool = S^T @ D @ S
        // Note: D is diagonal matrix of degrees. S^T D S is [B, K, K]

        // Batch Trace trick: sum(diagonal)

        // Degree Matrix D
        Tensor deg = adj.sum(new long[]{2}, false, new ScalarTypeOptional()); // [B, N]
        Tensor dMat = torch.diag_embed(deg); // [B, N, N]

        Tensor num = torch.einsum("bkk->b", new TensorVector(adjPool)); // Trace of A_pool
        Tensor den = torch.einsum("bkk->b", new TensorVector( s.transpose(1, 2).matmul(dMat).matmul(s)));

        Tensor cutLoss = num.div(den).neg().mean();

        // 4. Orthogonality Loss (Collapse regularization)
        // || S^T S / ||S||_F - I_k / sqrt(K) ||_F
        // Simplify: || S^T S - I ||
        Tensor sTs = s.transpose(1, 2).matmul(s);
        long K = s.size(2);
        Tensor iK = torch.eye(K, x.options()).unsqueeze(0);
        // Normalize S^T S is tricky, standard implementation:
        // || S^T S - I ||
        Tensor orthoLoss = sTs.div(sTs.norm()).sub(iK.div(new Scalar(Math.sqrt(K)))).norm().mean();

        return new Tensor[]{xPool, adjPool, cutLoss, orthoLoss, s};
    }
}