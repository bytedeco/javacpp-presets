package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * SortAggregation (SortPool-style).
 *
 * <p>For each group defined by {@code index}, sort members by the last feature
 * channel (descending), keep the top-{@code k} rows, zero-pad when a group has
 * fewer than {@code k} members, then flatten to {@code [dimSize, k * C]}.
 *
 * <p>Aligned with PyG {@code SortAggregation} / SortPooling for graph-level
 * pooling when {@code index} is a batch vector; also works as a neighborhood
 * aggregator when {@code index} is a target-node index.
 */
public class SortAggregation extends Aggregation {
    private final long k;

    public SortAggregation(long k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0, got " + k);
        }
        this.k = k;
    }

    public long k() {
        return k;
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // x: [E, C], index: [E]
        long C = x.size(1);

        // Pack neighbors into dense [N, maxDeg, C] with boolean mask.
        // fillValue=0 so padded slots contribute zeros after selection.
        Tensor[] packed = AggrUtils.to_dense_batch(x, index, dimSize, 0f);
        Tensor dense = packed[0]; // [N, L, C]
        Tensor mask = packed[1];  // [N, L] bool
        long L = dense.size(1);

        // Score = last feature channel; invalid (padded) positions get -inf so
        // they sort to the end under descending order.
        Tensor score = dense.select(2, C - 1).clone(); // [N, L]
        Tensor negInf = torch.full(score.shape(), new Scalar(Float.NEGATIVE_INFINITY), score.options());
        score = torch.where(mask, score, negInf);

        // torch.sort returns (values, indices); get1() is already the perm.
        T_TensorTensor_T sortRet = torch.sort(score, 1L, true);
        Tensor perm = sortRet.get1(); // [N, L]

        // Gather rows along degree dim using expanded permutation.
        Tensor permExp = perm.unsqueeze(2).expand(new long[]{dimSize, L, C});
        Tensor sorted = dense.gather(1, permExp); // [N, L, C]

        // Keep top-k (pad with zeros if L < k).
        Tensor topk;
        if (L >= k) {
            topk = sorted.slice(1, new LongOptional(0), new LongOptional(k), 1L); // [N, k, C]
        } else {
            Tensor pad = torch.zeros(new long[]{dimSize, k - L, C}, x.options());
            topk = torch.cat(new TensorVector(sorted, pad), 1); // [N, k, C]
        }

        // Flatten feature dim: [N, k * C]
        return topk.reshape(dimSize, k * C).contiguous();
    }
}
