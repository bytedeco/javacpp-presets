package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * Canonical scatter / degree / softmax backend for the geometric stack.
 *
 * <p>Isolated-node policy (PyG-aligned):
 * <ul>
 *   <li>sum / mean → 0</li>
 *   <li>max / min → 0 after reduce (sentinels cleared)</li>
 *   <li>mul / prod → 1 (identity of product)</li>
 * </ul>
 */
public final class AggrUtils {

    private AggrUtils() {}

    /** Normalize reduce aliases: add≡sum, mul≡prod. */
    public static String normalizeReduce(String reduce) {
        if (reduce == null) {
            throw new IllegalArgumentException("reduce must not be null");
        }
        switch (reduce) {
            case "add":
            case "sum":
                return "sum";
            case "mean":
                return "mean";
            case "max":
                return "max";
            case "min":
                return "min";
            case "mul":
            case "prod":
                return "prod";
            default:
                throw new UnsupportedOperationException(
                        "Unknown reduce='" + reduce + "' (supported: sum/add/mean/max/min/mul/prod)");
        }
    }

    /** Ensure 1-D Long index for index_add_ / index_reduce_ / index_select. */
    public static Tensor asLongIndex(Tensor index) {
        if (index == null) {
            throw new NullPointerException("index must not be null");
        }
        if (index.dim() != 1) {
            throw new IllegalArgumentException(
                    "index must be 1-D, got dim=" + index.dim() + " shape=" + shapeStr(index));
        }
        // scalar_type() may return a non-canonical proxy — always intern() both sides.
        if (index.scalar_type().intern() != torch.ScalarType.Long.intern()) {
            return index.to(torch.kLong());
        }
        return index;
    }

    /**
     * Scatter {@code src} [E, F...] into [dimSize, F...] along dim 0 using {@code index} [E].
     */
    public static Tensor scatter(Tensor src, Tensor index, long dimSize, String reduce) {
        if (src == null) {
            throw new NullPointerException("src must not be null");
        }
        index = asLongIndex(index);
        if (index.size(0) != src.size(0)) {
            throw new IllegalArgumentException(
                    "index.length (" + index.size(0) + ") != src.size(0) (" + src.size(0) + ")");
        }
        String r = normalizeReduce(reduce);
        if (dimSize < 0) {
            if (index.size(0) == 0) {
                dimSize = 0;
            } else {
                dimSize = index.max().item_long() + 1;
            }
        }

        long[] srcShape = src.shape();
        long[] outShape = new long[srcShape.length];
        outShape[0] = dimSize;
        System.arraycopy(srcShape, 1, outShape, 1, srcShape.length - 1);

        switch (r) {
            case "sum": {
                Tensor out = torch.zeros(outShape, src.options());
                return out.index_add_(0, index, src);
            }
            case "mean": {
                Tensor out = torch.zeros(outShape, src.options());
                Tensor sum = out.index_add_(0, index, src);
                Tensor count = torch.zeros(new long[]{dimSize}, src.options());
                Tensor ones = torch.ones(new long[]{src.size(0)}, src.options());
                count.index_add_(0, index, ones);
                count = count.clamp_min(new Scalar(1.0));
                for (int i = 1; i < outShape.length; i++) {
                    count = count.unsqueeze(i);
                }
                return sum.div(count);
            }
            case "max": {
                // include_self=false: only positions hit by index get reduced values.
                Tensor out = torch.full(outShape, new Scalar(-1.0e38), src.options());
                out = out.index_reduce_(0, index, src, "amax", false);
                return fillIsolated(out, index, dimSize, src.options());
            }
            case "min": {
                Tensor out = torch.full(outShape, new Scalar(1.0e38), src.options());
                out = out.index_reduce_(0, index, src, "amin", false);
                return fillIsolated(out, index, dimSize, src.options());
            }
            case "prod": {
                Tensor out = torch.ones(outShape, src.options());
                return out.index_reduce_(0, index, src, "prod", false);
            }
            default:
                throw new UnsupportedOperationException("Unknown reduce: " + r);
        }
    }

    /**
     * Zero-fill rows that received no messages (isolated nodes) after max/min reduce.
     * Vectorized: build presence mask via scatter of ones, broadcast to feature dims.
     */
    private static Tensor fillIsolated(Tensor out, Tensor index, long dimSize, TensorOptions opts) {
        if (dimSize == 0) {
            return out;
        }
        Tensor presence = torch.zeros(new long[]{dimSize}, opts);
        if (index.size(0) > 0) {
            Tensor ones = torch.ones(new long[]{index.size(0)}, opts);
            presence.index_add_(0, index, ones);
        }
        Tensor mask = presence.gt(new Scalar(0)); // [N]
        for (int i = 1; i < out.dim(); i++) {
            mask = mask.unsqueeze(i);
        }
        mask = mask.expand_as(out);
        return torch.where(mask, out, torch.zeros_like(out));
    }

    /**
     * Segment-wise softmax: for each group defined by {@code index},
     * {@code exp(x - max) / sum(exp(x - max))}. Used by attention and SoftmaxAggregation.
     */
    public static Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
        index = asLongIndex(index);
        // Numerically stable: x - max(x)
        // For max intermediate we need raw max, not zero-filled isolated — use internal path.
        long[] srcShape = src.shape();
        long[] outShape = new long[srcShape.length];
        outShape[0] = dimSize;
        System.arraycopy(srcShape, 1, outShape, 1, srcShape.length - 1);

        Tensor maxVal = torch.full(outShape, new Scalar(-1.0e38), src.options());
        maxVal = maxVal.index_reduce_(0, index, src, "amax", false);
        // Isolated stay at -1e38; they are never index_select'ed by real edges.

        Tensor maxExpanded = maxVal.index_select(0, index);
        Tensor num = src.sub(maxExpanded).exp();
        Tensor den = scatter(num, index, dimSize, "sum");
        Tensor denExpanded = den.index_select(0, index);
        return num.div(denExpanded.add(new Scalar(1e-12)));
    }

    /** Node degree from edge endpoint index [E] → [dimSize]. */
    public static Tensor compute_degree(Tensor index, long dimSize) {
        index = asLongIndex(index);
        Tensor ones = torch.ones(
                new long[]{index.size(0)},
                index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
        Tensor out = torch.zeros(
                new long[]{dimSize},
                index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
        return out.index_add_(0, index, ones);
    }

    /**
     * Pack sparse neighbor features into a dense batch for Median/Quantile/LSTM aggregators.
     *
     * @return {dense_x [N, MaxDeg, F], mask [N, MaxDeg], lengths [N]}
     */
    public static Tensor[] to_dense_batch(Tensor x, Tensor index, long dimSize, float fillValue) {
        long numEdges = x.size(0);
        long numFeatures = x.size(1);
        index = asLongIndex(index);

        Tensor lengths = compute_degree(index, dimSize).to(torch.ScalarType.Long);
        long maxDeg = lengths.max().item().toLong();
        if (maxDeg == 0) {
            Tensor dense = torch.full(new long[]{dimSize, 1, numFeatures}, new Scalar(fillValue), x.options());
            Tensor mask = torch.zeros(new long[]{dimSize, 1},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Bool)));
            return new Tensor[]{dense, mask, lengths};
        }

        T_TensorTensor_T sortRet = torch.sort(index);
        Tensor perm = sortRet.get1();
        Tensor sortedIndex = sortRet.get0();
        Tensor sortedX = x.index_select(0, perm);

        Tensor endPos = torch.cumsum(lengths, 0);
        Tensor startPos = torch.cat(new TensorVector(
                torch.zeros(new long[]{1}, lengths.options()),
                endPos.slice(0, new LongOptional(0), new LongOptional(dimSize - 1), 1L)), 0);

        Tensor edgeStartPos = startPos.index_select(0, sortedIndex);
        Tensor range = torch.arange(new Scalar(numEdges), index.options());
        Tensor innerIdx = range.sub(edgeStartPos);

        Tensor dense = torch.full(new long[]{dimSize, maxDeg, numFeatures}, new Scalar(fillValue), x.options());
        Tensor flatIdx = sortedIndex.mul(new Scalar(maxDeg)).add(innerIdx);
        Tensor denseFlat = dense.view(dimSize * maxDeg, numFeatures);
        denseFlat.index_copy_(0, flatIdx, sortedX);
        dense = denseFlat.view(dimSize, maxDeg, numFeatures);

        Tensor degRange = torch.arange(new Scalar(maxDeg), lengths.options()).unsqueeze(0);
        Tensor mask = degRange.lt(lengths.unsqueeze(1));
        return new Tensor[]{dense, mask, lengths};
    }

    private static String shapeStr(Tensor t) {
        long[] s = t.shape();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < s.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(s[i]);
        }
        return sb.append(']').toString();
    }
}
