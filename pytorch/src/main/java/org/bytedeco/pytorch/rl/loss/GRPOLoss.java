package org.bytedeco.pytorch.rl.loss;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Distribution;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Group-Relative Policy Optimization loss (DeepSeek-R1 style, clipped).
 *
 * <p>For each prompt, {@code G} completions are scored; advantages are
 * group-normalized. A PPO-style clip is applied on the importance ratio.
 *
 * <p><b>Discrete note:</b> {@code Categorical.log_prob} already returns
 * {@code [B*G]} — do <em>not</em> {@code sum(-1)} or the batch collapses to a scalar.
 */
public class GRPOLoss {
    public static Tensor computeLoss(Distribution currDist, Tensor actions, Tensor oldLogProbs,
                                     Tensor groupRewards, float clipEps) {
        // groupRewards: [Batch, GroupSize]
        Tensor mean = groupRewards.mean(new long[]{-1}, true, new ScalarTypeOptional());
        // Population std via mean of squares — avoids df<=0 on tiny G
        Tensor centered = groupRewards.sub(mean);
        Tensor std = centered.pow(new Scalar(2.0)).mean(new long[]{-1}, true, new ScalarTypeOptional())
                .sqrt().add(new Scalar(1e-8));
        Tensor advantages = centered.div(std).flatten(); // [Batch * GroupSize]

        Tensor logProbs = currDist.log_prob(actions);
        if (logProbs.dim() > 1) {
            logProbs = logProbs.sum(-1); // continuous / multi-dim only
        }
        Tensor oldLp = oldLogProbs;
        while (oldLp.dim() > 1 && oldLp.size(oldLp.dim() - 1) == 1) {
            oldLp = oldLp.squeeze(oldLp.dim() - 1);
        }
        Tensor ratio = exp(logProbs.sub(oldLp));

        Tensor surr1 = ratio.mul(advantages);
        Tensor surr2 = clamp(ratio,
                new ScalarOptional(new Scalar(1.0 - clipEps)),
                new ScalarOptional(new Scalar(1.0 + clipEps))).mul(advantages);

        return min(surr1, surr2).mean().neg();
    }
}
