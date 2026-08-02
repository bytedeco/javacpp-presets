package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.transforms.DistributionTransform;
import org.bytedeco.pytorch.distribution.transforms.DistributionTransforms;
import org.bytedeco.pytorch.global.torch;

/**
 * TransformedDistribution — push-forward of a base distribution through an
 * invertible transform T (mirrors {@code torch.distributions.TransformedDistribution}).
 *
 * <p>If X ~ P_base and Y = T(X), then
 * <pre>
 *   log p_Y(y) = log p_X(T⁻¹(y)) − log|det J_T(T⁻¹(y))|
 *   sample:    Y = T(X), X ~ P_base
 *   H(Y)       = H(X) + E_X[log|det J_T(X)|]   (Monte-Carlo when needed)
 * </pre>
 *
 * <p>Supports a single transform or a composition via {@link DistributionTransforms#compose}.
 * Typical uses:
 * <ul>
 *   <li>LogNormal ≈ TransformedDistribution(Normal(μ,σ), ExpTransform)</li>
 *   <li>SAC squashed Gaussian ≈ TransformedDistribution(Normal, TanhTransform)</li>
 *   <li>Affine reparameterization of any location-scale family</li>
 * </ul>
 */
public class TransformedDistribution extends Distribution implements AutoCloseable {

    private final Distribution baseDist;
    private final DistributionTransform transform;
    private final boolean ownsBase;
    private final boolean ownsTransform;
    private boolean closed = false;

    /**
     * @param baseDist  base distribution P_X (not null)
     * @param transform invertible transform T (not null)
     */
    public TransformedDistribution(Distribution baseDist, DistributionTransform transform) {
        this(baseDist, transform, false, false);
    }

    /**
     * @param ownsBase      if true, {@link #close()} will close the base dist
     * @param ownsTransform if true, {@link #close()} will close the transform
     */
    public TransformedDistribution(Distribution baseDist, DistributionTransform transform,
                                   boolean ownsBase, boolean ownsTransform) {
        if (baseDist == null) {
            throw new IllegalArgumentException("baseDist cannot be null");
        }
        if (transform == null) {
            throw new IllegalArgumentException("transform cannot be null");
        }
        this.baseDist = baseDist;
        this.transform = transform;
        this.ownsBase = ownsBase;
        this.ownsTransform = ownsTransform;
    }

    /** Convenience: base + a composition of transforms (applied left-to-right). */
    public TransformedDistribution(Distribution baseDist, DistributionTransform... transforms) {
        this(baseDist,
                transforms.length == 1 ? transforms[0] : DistributionTransforms.compose(transforms),
                false,
                transforms.length != 1 /* compose is owned */);
    }

    @Override
    public String name() {
        return "Transformed(" + baseDist.name() + ")";
    }

    public Distribution getBaseDist() {
        return baseDist;
    }

    public DistributionTransform getTransform() {
        return transform;
    }

    /**
     * Sample Y = T(X), X ~ base.
     */
    @Override
    public Tensor sample(long... sampleShape) {
        checkOpen();
        Tensor x = baseDist.sample(sampleShape);
        Tensor y = transform.forward(x);
        x.close();
        return y;
    }

    /**
     * log p_Y(y) = log p_X(T⁻¹(y)) − log|det J_T(T⁻¹(y))|.
     *
     * <p>When the transform has {@code eventDim > 0}, the jacobian is reduced over
     * those event dimensions so that the result shape matches {@code base.log_prob}.</p>
     */
    @Override
    public Tensor log_prob(Tensor y) {
        checkOpen();
        if (y == null || y.numel() == 0) {
            throw new IllegalArgumentException("log_prob input cannot be null/empty");
        }

        Tensor x = transform.inverse(y);
        Tensor baseLogProb = baseDist.log_prob(x);
        Tensor logDet = transform.logAbsDetJacobian(x, y);

        // Reduce jacobian over transform event dims if needed so shapes align
        int eventDim = transform.eventDim();
        Tensor reducedLogDet = logDet;
        if (eventDim > 0 && logDet.dim() >= eventDim) {
            long[] dims = new long[eventDim];
            for (int i = 0; i < eventDim; i++) {
                dims[i] = logDet.dim() - eventDim + i;
            }
            reducedLogDet = logDet.sum(dims, false, new ScalarTypeOptional());
            if (reducedLogDet != logDet) {
                logDet.close();
            }
        }

        // Also reduce base log_prob over same event dims if shapes still differ
        Tensor reducedBase = baseLogProb;
        if (eventDim > 0 && baseLogProb.dim() > reducedLogDet.dim()) {
            long reduceCount = baseLogProb.dim() - reducedLogDet.dim();
            if (reduceCount > 0) {
                long[] dims = new long[(int) reduceCount];
                for (int i = 0; i < reduceCount; i++) {
                    dims[i] = baseLogProb.dim() - reduceCount + i;
                }
                Tensor summed = baseLogProb.sum(dims, false, new ScalarTypeOptional());
                baseLogProb.close();
                reducedBase = summed;
            }
        }

        Tensor logProb = reducedBase.sub(reducedLogDet);

        x.close();
        reducedBase.close();
        reducedLogDet.close();

        return logProb;
    }

    /**
     * Mean of Y. Default: linear approximation T(E[X]).
     * For nonlinear transforms prefer {@link #mean(boolean) mean(true)} (Monte-Carlo).
     */
    @Override
    public Tensor mean() {
        return mean(false);
    }

    /**
     * @param useMonteCarlo if true, estimate E[T(X)] with 4096 base samples;
     *                      if false, return T(E[X]) (exact only for affine T).
     */
    public Tensor mean(boolean useMonteCarlo) {
        checkOpen();
        if (!useMonteCarlo) {
            Tensor baseMean = baseDist.mean();
            Tensor y = transform.forward(baseMean);
            baseMean.close();
            return y;
        }
        final long n = 4096;
        Tensor samples = baseDist.sample(n);
        Tensor transformed = transform.forward(samples);
        // mean over sample dim 0
        Tensor mcMean = transformed.mean(new long[]{0}, false, new ScalarTypeOptional());
        samples.close();
        transformed.close();
        return mcMean;
    }

    /**
     * Entropy H(Y) = H(X) + E_X[log|det J_T(X)|].
     * Expectation estimated by Monte-Carlo (1024 samples).
     * Exact for volume-preserving / constant-jacobian transforms.
     */
    @Override
    public Tensor entropy() {
        checkOpen();
        Tensor baseEntropy = baseDist.entropy();

        final long n = 1024;
        Tensor baseSamples = baseDist.sample(n);
        Tensor transformed = transform.forward(baseSamples);
        Tensor logDet = transform.logAbsDetJacobian(baseSamples, transformed);

        // mean over sample dim 0 (and any event dims of the jacobian)
        Tensor meanLogDet;
        if (logDet.dim() == 0) {
            meanLogDet = logDet.clone();
        } else {
            // average over leading sample dimension
            meanLogDet = logDet.mean(new long[]{0}, false, new ScalarTypeOptional());
            // if still multi-dim (event), sum event dims for scalar entropy per batch
            if (meanLogDet.dim() > baseEntropy.dim()) {
                long[] rest = new long[(int) (meanLogDet.dim() - Math.max(baseEntropy.dim(), 0))];
                for (int i = 0; i < rest.length; i++) {
                    rest[i] = baseEntropy.dim() + i;
                }
                if (rest.length > 0) {
                    Tensor summed = meanLogDet.sum(rest, false, new ScalarTypeOptional());
                    meanLogDet.close();
                    meanLogDet = summed;
                }
            }
        }

        Tensor ent = baseEntropy.add(meanLogDet);

        baseEntropy.close();
        baseSamples.close();
        transformed.close();
        logDet.close();
        meanLogDet.close();

        return ent;
    }

    /**
     * Debug helper: check T⁻¹(T(x)) ≈ x within atol.
     */
    public boolean validateInvertibility(Tensor testX, double atol) {
        checkOpen();
        Tensor y = transform.forward(testX);
        Tensor x2 = transform.inverse(y);
        Tensor diff = torch.abs(x2.sub(testX));
        boolean ok = torch.all(torch.lt(diff, new Scalar(atol))).item().toBool();
        y.close();
        x2.close();
        diff.close();
        return ok;
    }

    public boolean validateInvertibility(Tensor testX) {
        return validateInvertibility(testX, 1e-5);
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("TransformedDistribution already closed");
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (ownsBase && baseDist instanceof AutoCloseable) {
            try {
                ((AutoCloseable) baseDist).close();
            } catch (Exception e) {
                System.err.println("Failed to close baseDist: " + e.getMessage());
            }
        }
        if (ownsTransform) {
            try {
                transform.close();
            } catch (Exception e) {
                System.err.println("Failed to close transform: " + e.getMessage());
            }
        }
    }
}
