package org.bytedeco.pytorch.geometric.nn.norm;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

/**
 * PairNorm (Zhao & Akoglu): center then rescale to a fixed total/per-channel RMS.
 *
 * <pre>
 *   x_c = x − mean(x)                 // per graph if batch given
 *   x'  = s · x_c / (RMS(x_c) + ε)
 * </pre>
 * When {@code scaleIndividually=true}, RMS is per-channel; otherwise a single
 * scalar RMS is used (all features share one scale).
 */
public class PairNorm extends Module {

    private final double scale;
    private final boolean scaleIndividually;
    private final double eps;

    public PairNorm() {
        this(1.0, false, 1e-6);
    }

    public PairNorm(double scale, boolean scaleIndividually) {
        this(scale, scaleIndividually, 1e-6);
    }

    public PairNorm(double scale, boolean scaleIndividually, double eps) {
        super();
        this.scale = scale;
        this.scaleIndividually = scaleIndividually;
        this.eps = eps;
    }

    /** Single-graph convenience. */
    public Tensor forward(Tensor x) {
        return forward(x, (Tensor) null);
    }

    /**
     * @param x     [N, C]
     * @param batch [N] long graph ids, or null
     */
    public Tensor forward(Tensor x, Tensor batch) {
        if (x == null || x.dim() != 2) {
            throw new IllegalArgumentException("x must be [N, C]");
        }

        Tensor xCentered;
        Tensor rootMeanSq;

        if (batch == null) {
            xCentered = x.sub(x.mean(new long[]{0}, true, new ScalarTypeOptional(torch.ScalarType.Float)));
            Tensor sq = xCentered.pow(new Scalar(2));
            if (scaleIndividually) {
                rootMeanSq = sq.mean(new long[]{0}, true, new ScalarTypeOptional(torch.ScalarType.Float)).sqrt();
            } else {
                rootMeanSq = sq.mean().sqrt(); // scalar
            }
        } else {
            batch = AggrUtils.asLongIndex(batch);
            long numGraphs = batch.size(0) == 0 ? 1 : batch.max().item_long() + 1;
            Tensor mean = AggrUtils.scatter(x, batch, numGraphs, "mean");
            xCentered = x.sub(mean.index_select(0, batch));
            Tensor sq = xCentered.pow(new Scalar(2));

            if (scaleIndividually) {
                Tensor graphMeanSq = AggrUtils.scatter(sq, batch, numGraphs, "mean"); // [G,C]
                rootMeanSq = graphMeanSq.sqrt().index_select(0, batch);
            } else {
                // Per-graph mean of all squared entries: mean over C first → [N], then scatter
                Tensor rowMeanSq = sq.mean(new long[]{1}, false, new ScalarTypeOptional(torch.ScalarType.Float));
                Tensor graphRms = AggrUtils.scatter(rowMeanSq.unsqueeze(1), batch, numGraphs, "mean")
                        .sqrt(); // [G,1]
                rootMeanSq = graphRms.index_select(0, batch);
            }
        }

        return xCentered.div(rootMeanSq.add(new Scalar(eps))).mul(new Scalar(scale));
    }

    public double getScale() {
        return scale;
    }

    public boolean isScaleIndividually() {
        return scaleIndividually;
    }
}
