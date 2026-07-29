/*
 * Ported from torch-rechub-scala: torchrec/basic/MetaOptimizer.scala
 *
 * MetaBalance - scales gradients and balances gradient across tasks.
 * Multi-task gradient balancing helper (no retain_graph support in JavaCPP).
 */
package org.bytedeco.pytorch.utils.recommend.basic;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * MetaBalance - scales gradients and balances gradient across tasks.
 *
 * @param relaxFactor Relaxation factor for gradient scaling (default: 0.7, range: [0, 1))
 * @param beta Moving average coefficient (default: 0.9, range: [0, 1))
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MetaBalance {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final float relaxFactor;
    private final float beta;

    // Per-parameter gradient norms history
    private final Map<Integer, List<Float>> gradNorms = new HashMap<>();
    // Per-parameter accumulated gradient sum
    private final Map<Integer, Tensor> sumGradients = new HashMap<>();
    // Per-parameter first gradient flag
    private final Map<Integer, Boolean> firstGradient = new HashMap<>();

    public MetaBalance() {
        this(0.7f, 0.9f);
    }

    public MetaBalance(float relaxFactor, float beta) {
        if (relaxFactor < 0.0f || relaxFactor >= 1.0f) {
            throw new IllegalArgumentException(
                    "Invalid relax_factor: " + relaxFactor + ", it should be 0. <= relax_factor < 1.");
        }
        if (beta < 0.0f || beta >= 1.0f) {
            throw new IllegalArgumentException(
                    "Invalid beta: " + beta + ", it should be 0. <= beta < 1.");
        }
        this.relaxFactor = relaxFactor;
        this.beta = beta;
    }

    /**
     * Compute and accumulate scaled gradients for multiple task losses.
     * Note: JavaCPP backward() does not support retain_graph; losses are computed
     * and their gradients accumulated sequentially.
     */
    public void step(List<Tensor> params, List<Tensor> losses) {
        if (losses == null || losses.isEmpty()) {
            throw new IllegalArgumentException("At least one loss must be provided");
        }
        if (params == null || params.isEmpty()) {
            throw new IllegalArgumentException("At least one parameter must be provided");
        }

        int numTasks = losses.size();

        // Initialize state for each parameter
        for (Tensor param : params) {
            int paramId = System.identityHashCode(param);
            if (!gradNorms.containsKey(paramId)) {
                List<Float> norms = new ArrayList<>(numTasks);
                for (int i = 0; i < numTasks; i++) {
                    norms.add(0.0f);
                }
                gradNorms.put(paramId, norms);
            }
            if (!sumGradients.containsKey(paramId)) {
                Tensor zero = param.clone();
                zero.zero_();
                sumGradients.put(paramId, zero);
            }
            if (!firstGradient.containsKey(paramId)) {
                firstGradient.put(paramId, true);
            }
        }

        for (Tensor loss : losses) {
            loss.backward();

            for (Tensor param : params) {
                int paramId = System.identityHashCode(param);
                Tensor grad = param.grad();
                if (grad == null) {
                    return;
                }
                if (grad.is_sparse()) {
                    throw new RuntimeException("MetaBalance does not support sparse gradients");
                }

                List<Float> norms = gradNorms.get(paramId);
                Tensor sumGrad = sumGradients.get(paramId);

                float gradNorm = grad.norm().item().toFloat();

                // Update moving average of gradient norm
                norms.set(0, norms.get(0) * beta + (1.0f - beta) * gradNorm);

                // Scale: scaled = grad * (norms[0] / (norms[0] + 1e-5)) * relax + grad * (1 - relax)
                float scale = (norms.get(0) / (norms.get(0) + 1e-5f)) * relaxFactor;
                Tensor scaledGrad = grad.mul(new Scalar(scale)).add(grad.mul(new Scalar(1.0f - relaxFactor)));

                sumGrad.add_(scaledGrad);
                grad.zero_();
            }

            loss.close();
        }
    }

    /** Apply the accumulated gradients to parameters. */
    public void applyGradients(List<Tensor> params) {
        for (Tensor param : params) {
            int paramId = System.identityHashCode(param);
            Tensor sumGrad = sumGradients.get(paramId);
            if (sumGrad != null) {
                param.grad().copy_(sumGrad);
                sumGrad.zero_();
            }
        }
    }

    /** Reset the optimizer state. */
    public void reset() {
        for (List<Float> norms : gradNorms.values()) {
            for (int i = 0; i < norms.size(); i++) {
                norms.set(i, 0.0f);
            }
        }
        for (Tensor g : sumGradients.values()) {
            g.zero_();
        }
        for (Integer k : firstGradient.keySet()) {
            firstGradient.put(k, true);
        }
    }
}
