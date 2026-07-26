package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;

/**
 * KNN Index 基类
 * 用于查找 k-Nearest Neighbors
 */
public abstract class KNNIndex {
    protected long k;

    public KNNIndex(long k) {
        this.k = k;
    }

    /**
     * @param x Source points (Queries) [N, D]
     * @param y Target points (Database) [M, D] (Optional, if null then y=x)
     * @param batchX Batch indices for x [N] (Optional)
     * @param batchY Batch indices for y [M] (Optional)
     * @return Tensor pair: (distances, indices) -> indices shape [N, k]
     */
    public abstract Tensor[] search(Tensor x, Tensor y, Tensor batchX, Tensor batchY);
}