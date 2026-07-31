/*
 * Duration-aware deconfounding head (D2Q-style).
 *
 * Production reference:
 *   Zhan et al., "Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation",
 *   KDD 2022 (Kuaishou). Industrial watch-time ranking often confounds user interest with
 *   video length; D2Q learns quantile-conditioned watch-time so longer videos are not
 *   systematically preferred solely due to length.
 *
 * Design (faithful to the paper's core idea, simplified for library use):
 *   1. Bucket video duration into K quantiles (or fixed buckets).
 *   2. Condition the watch-time regressor on duration bucket embedding.
 *   3. Optionally emit both raw watch-time prediction and deconfounded interest score.
 *
 * Inputs:
 *   userItemEmb  [B, D]  — fused user/item representation from backbone
 *   durationBucket [B]   — long indices in [0, numBuckets)
 * Outputs:
 *   Tensor of shape [B, 2]: [watch_time_pred, interest_score]
 */
package org.bytedeco.pytorch.recommend.basic.layers.industry;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DurationDeconfoundHead extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingImpl durationEmbed;
    private final MLP watchTimeTower;
    private final MLP interestTower;
    private final int durationEmbedDim;

    public DurationDeconfoundHead(int inputDim, int numDurationBuckets) {
        this(inputDim, numDurationBuckets, 16, new long[]{128L, 64L}, DeviceSupport.backend());
    }

    public DurationDeconfoundHead(int inputDim, int numDurationBuckets, int durationEmbedDim,
                                  long[] hiddenDims, String device) {
        super("DurationDeconfoundHead");
        if (numDurationBuckets < 2) {
            throw new IllegalArgumentException("numDurationBuckets must be >= 2");
        }
        this.durationEmbedDim = durationEmbedDim;

        EmbeddingOptions opts = new EmbeddingOptions(numDurationBuckets, durationEmbedDim);
        this.durationEmbed = new EmbeddingImpl(opts);
        register_module("duration_embed", durationEmbed);

        long towerIn = (long) inputDim + durationEmbedDim;
        this.watchTimeTower = new MLP(towerIn, hiddenDims, 1L, "relu", 0.1f, false, false, true, device);
        // interest score is intentionally NOT conditioned on duration (deconfounded)
        this.interestTower = new MLP(inputDim, hiddenDims, 1L, "relu", 0.1f, false, false, true, device);
        register_module("watch_time_tower", watchTimeTower);
        register_module("interest_tower", interestTower);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            durationEmbed.to(dev, false);
        }
    }

    /**
     * @param userItemEmb    [B, D]
     * @param durationBucket [B] long
     * @return [B, 2] columns: watch_time (softplus), interest (sigmoid)
     */
    public Tensor forward(Tensor userItemEmb, Tensor durationBucket) {
        Tensor dEmb = durationEmbed.forward(durationBucket.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long));
        TensorVector cat = new TensorVector();
        cat.push_back(userItemEmb);
        cat.push_back(dEmb);
        Tensor cond = torch.cat(cat, 1L);

        Tensor wt = torch.softplus(watchTimeTower.forward(cond).squeeze(1L));
        Tensor interest = interestTower.forward(userItemEmb).squeeze(1L).sigmoid();

        TensorVector out = new TensorVector();
        out.push_back(wt.unsqueeze(1L));
        out.push_back(interest.unsqueeze(1L));
        return torch.cat(out, 1L);
    }

    public int durationEmbedDim() {
        return durationEmbedDim;
    }
}
