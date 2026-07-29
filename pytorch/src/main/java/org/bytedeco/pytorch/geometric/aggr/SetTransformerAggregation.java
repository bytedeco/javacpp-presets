package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Set Transformer Aggregation (PMA-style).
 *
 * <p>{@code numSeeds == 1}: efficient scatter-softmax attention over neighbors.
 * <p>{@code numSeeds  > 1}: pack neighbors via {@link AggrUtils#to_dense_batch},
 * run multi-seed attention per group, then mean-pool seeds → {@code [dimSize, outChannels]}.
 */
public class SetTransformerAggregation extends Aggregation {
    private final Tensor seed;
    private final LinearImpl linQ, linK, linV, linOut;
    private final long numHeads;
    private final long outChannels;
    private final long numSeeds;

    public SetTransformerAggregation(long inChannels, long outChannels, long numHeads, long numSeeds) {
        if (numSeeds <= 0) {
            throw new IllegalArgumentException("numSeeds must be > 0, got " + numSeeds);
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be > 0, got " + numHeads);
        }
        if (outChannels % numHeads != 0) {
            throw new IllegalArgumentException(
                    "outChannels (" + outChannels + ") must be divisible by numHeads (" + numHeads + ")");
        }
        this.numHeads = numHeads;
        this.outChannels = outChannels;
        this.numSeeds = numSeeds;

        // Keep a strong handle: register_parameter is ByRef and must not drop the leaf.
        this.seed = torch.randn(new long[]{numSeeds, outChannels}).clone();
        register_parameter("seed", seed);

        this.linQ = new LinearImpl(outChannels, outChannels);
        this.linK = new LinearImpl(inChannels, outChannels);
        this.linV = new LinearImpl(inChannels, outChannels);
        this.linOut = new LinearImpl(outChannels, outChannels);

        register_module("linQ", linQ);
        register_module("linK", linK);
        register_module("linV", linV);
        register_module("linOut", linOut);
    }

    public long numSeeds() {
        return numSeeds;
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        if (numSeeds == 1) {
            return forwardSingleSeed(x, index, dimSize);
        }
        return forwardMultiSeed(x, index, dimSize);
    }

    /** Scatter-softmax attention with a single learnable seed (local / global). */
    private Tensor forwardSingleSeed(Tensor x, Tensor index, long dimSize) {
        Tensor Q = linQ.forward(seed);          // [1, Out]
        Tensor K = linK.forward(x);             // [E, Out]
        Tensor V = linV.forward(x);             // [E, Out]

        Tensor scores = K.matmul(Q.t());        // [E, 1]
        scores = scores.mul(new Scalar(1.0 / Math.sqrt(outChannels)));
        Tensor alpha = AggrUtils.scatter_softmax(scores, index, dimSize);
        Tensor agg = AggrUtils.scatter(V.mul(alpha), index, dimSize, "sum");
        return linOut.forward(agg);
    }

    /**
     * Multi-seed PMA over dense-packed neighborhoods.
     * Output is mean-pooled over seeds → [dimSize, outChannels].
     */
    private Tensor forwardMultiSeed(Tensor x, Tensor index, long dimSize) {
        Tensor[] packed = AggrUtils.to_dense_batch(x, index, dimSize, 0f);
        Tensor denseX = packed[0]; // [N, L, Fin]
        Tensor mask = packed[1];   // [N, L] bool
        long L = denseX.size(1);
        long H = numHeads;
        long D = outChannels / H;

        // Project keys/values from node features; project queries from seeds.
        // Flatten N*L for linear, then reshape back.
        Tensor flatX = denseX.reshape(dimSize * L, denseX.size(2));
        Tensor K = linK.forward(flatX).view(dimSize, L, H, D).transpose(1, 2); // [N, H, L, D]
        Tensor V = linV.forward(flatX).view(dimSize, L, H, D).transpose(1, 2); // [N, H, L, D]
        Tensor Q = linQ.forward(seed)                                          // [S, Out]
                .view(numSeeds, H, D)
                .unsqueeze(0)
                .expand(new long[]{dimSize, numSeeds, H, D})
                .transpose(1, 2);                                              // [N, H, S, D]

        // scores: [N, H, S, L]
        Tensor scores = Q.matmul(K.transpose(-2, -1))
                .mul(new Scalar(1.0 / Math.sqrt(D)));

        // Mask padded positions with -inf before softmax over L.
        Tensor maskExp = mask.unsqueeze(1).unsqueeze(2); // [N, 1, 1, L]
        Tensor negInf = torch.full(scores.shape(), new Scalar(Float.NEGATIVE_INFINITY), scores.options());
        scores = torch.where(maskExp, scores, negInf);

        // Groups with zero degree: mask is all-false → softmax of -inf is NaN.
        // Replace all-false rows with zeros after softmax.
        Tensor alpha = torch.softmax(scores, -1); // [N, H, S, L]
        // any(dim, keepdim) — no ScalarTypeOptional overload
        Tensor valid = mask.any(new long[]{1}, true); // [N, 1]
        alpha = torch.where(valid.unsqueeze(1).unsqueeze(2), alpha, torch.zeros_like(alpha));

        // attn @ V → [N, H, S, D] → [N, S, Out]
        Tensor out = alpha.matmul(V)                         // [N, H, S, D]
                .transpose(1, 2)                             // [N, S, H, D]
                .contiguous()
                .view(dimSize, numSeeds, outChannels);

        // Mean-pool seeds (PyG often flattens; mean keeps [N, Out] for drop-in use).
        Tensor pooled = out.mean(new long[]{1}, false,
                new ScalarTypeOptional(torch.ScalarType.Float)); // [N, Out]
        return linOut.forward(pooled);
    }
}
