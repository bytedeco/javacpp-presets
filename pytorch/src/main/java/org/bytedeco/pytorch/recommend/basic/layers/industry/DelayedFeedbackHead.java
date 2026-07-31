/*
 * Delayed Feedback Model (DFM) calibration head — e-commerce / ads conversion.
 *
 * Production / paper references:
 *   - Chapelle, "Modeling Delayed Feedback in Display Advertising", KDD 2014 (Criteo)
 *   - Yasui et al., "A Feedback Shift Correction in Predicting Conversion Rates
 *     under Delayed Feedback", WWW 2020
 *   - Alibaba / ByteDance industrial CVR pipelines routinely correct for
 *     right-censored conversions that arrive hours–days after click.
 *
 * Core idea (Chapelle DFM):
 *   Observed conversion y depends on whether conversion delay d is less than
 *   the elapsed time e since click. Likelihood factors into:
 *     P(click converts eventually) * P(delay <= e | converts)
 *   We model p_cvr with a tower and p(delay) with an exponential / log-normal head.
 *
 * This module exposes:
 *   - cvr logits / probability
 *   - expected delay (hours)
 *   - delayed-feedback-aware training loss helper
 */
package org.bytedeco.pytorch.recommend.basic.layers.industry;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DelayedFeedbackHead extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final MLP cvrTower;
    private final MLP delayTower; // predicts log(lambda) for Exp(lambda) delay, or mu of log-normal

    public DelayedFeedbackHead(int inputDim) {
        this(inputDim, new long[]{128L, 64L}, DeviceSupport.backend());
    }

    public DelayedFeedbackHead(int inputDim, long[] hiddenDims, String device) {
        super("DelayedFeedbackHead");
        this.cvrTower = new MLP(inputDim, hiddenDims, 1L, "relu", 0.1f, false, false, true, device);
        this.delayTower = new MLP(inputDim, hiddenDims, 1L, "relu", 0.1f, false, false, true, device);
        register_module("cvr_tower", cvrTower);
        register_module("delay_tower", delayTower);
    }

    /** @return cvr probability [B] (sigmoid) */
    public Tensor predictCvr(Tensor x) {
        return cvrTower.forward(x).squeeze(1L).sigmoid();
    }

    /**
     * Exponential delay rate lambda = softplus(f(x)).
     * E[delay] = 1/lambda.
     * @return lambda [B]
     */
    public Tensor predictDelayRate(Tensor x) {
        return torch.softplus(delayTower.forward(x).squeeze(1L))
                .add(new Scalar(1e-6f));
    }

    /**
     * Chapelle-style negative log-likelihood for right-censored conversions.
     *
     * @param x            feature representation [B, D]
     * @param converted    1 if conversion observed before training time, else 0  [B]
     * @param elapsedHours hours since click (censoring time if not converted)   [B]
     * @return scalar mean NLL
     */
    public Tensor delayedFeedbackNll(Tensor x, Tensor converted, Tensor elapsedHours) {
        Tensor p = predictCvr(x).clamp(new ScalarOptional(new Scalar(1e-6f)), new ScalarOptional(new Scalar(1.0f - 1e-6f)));
        Tensor lambda = predictDelayRate(x);
        Tensor y = converted.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor e = elapsedHours.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float)
                .clamp_min(new Scalar(1e-6f));

        // For converted samples: -log(p * lambda * exp(-lambda * e))
        //   = -log(p) - log(lambda) + lambda * e
        Tensor nllPos = p.log().neg()
                .add(lambda.log().neg())
                .add(lambda.mul(e));

        // For censored (not yet converted): -log(1 - p + p * exp(-lambda * e))
        // survival of conversion-by-e under mixture
        Tensor survival = torch.exp(lambda.mul(e).neg());
        Tensor pCensored = torch.sub(torch.ones_like(p), p).add(p.mul(survival));
        Tensor nllNeg = pCensored.clamp_min(new Scalar(1e-6f)).log().neg();

        Tensor nll = nllPos.mul(y).add(nllNeg.mul(torch.sub(torch.ones_like(y), y)));
        return nll.mean();
    }

    /** Convenience: returns [cvr_prob, expected_delay_hours] as [B, 2]. */
    public Tensor forward(Tensor x) {
        Tensor p = predictCvr(x);
        Tensor lambda = predictDelayRate(x);
        Tensor expDelay = torch.div(torch.ones_like(lambda), lambda);
        org.bytedeco.pytorch.TensorVector out = new org.bytedeco.pytorch.TensorVector();
        out.push_back(p.unsqueeze(1L));
        out.push_back(expDelay.unsqueeze(1L));
        return torch.cat(out, 1L);
    }
}
