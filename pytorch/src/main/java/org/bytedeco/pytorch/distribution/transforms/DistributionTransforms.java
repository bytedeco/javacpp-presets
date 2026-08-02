package org.bytedeco.pytorch.distribution.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

/**
 * Common invertible transforms used with
 * {@link org.bytedeco.pytorch.distribution.TransformedDistribution}.
 *
 * <p>Factory methods return concrete bijective (or half-bijective) maps that
 * mirror {@code torch.distributions.transforms}.</p>
 */
public final class DistributionTransforms {
    private DistributionTransforms() {}

    /** Identity: Y = X. */
    public static DistributionTransform identity() {
        return new IdentityTransform();
    }

    /** Exp: Y = exp(X), domain ℝ → (0, ∞). Used by LogNormal. */
    public static DistributionTransform exp() {
        return new ExpTransform();
    }

    /** Affine: Y = loc + scale * X (scale ≠ 0). */
    public static DistributionTransform affine(Tensor loc, Tensor scale) {
        return new AffineTransform(loc, scale);
    }

    /** Sigmoid: Y = 1/(1+exp(-X)), domain ℝ → (0,1). */
    public static DistributionTransform sigmoid() {
        return new SigmoidTransform();
    }

    /** Softplus: Y = log(1+exp(X)), domain ℝ → (0,∞). */
    public static DistributionTransform softplus() {
        return new SoftplusTransform();
    }

    /** Abs: Y = |X| (not bijective; inverse returns +Y). Used by HalfNormal. */
    public static DistributionTransform abs() {
        return new AbsTransform();
    }

    /** Tanh: Y = tanh(X), domain ℝ → (-1,1). Common in SAC squashing. */
    public static DistributionTransform tanh() {
        return new TanhTransform();
    }

    /** Compose transforms left-to-right: T_{n-1} ∘ … ∘ T_0. */
    public static DistributionTransform compose(DistributionTransform... transforms) {
        return new ComposeTransform(transforms);
    }

    // ------------------------------------------------------------------
    // Implementations
    // ------------------------------------------------------------------

    public static final class IdentityTransform extends DistributionTransform {
        @Override public int eventDim() { return 0; }

        @Override
        public Tensor forward(Tensor x) {
            return x.clone();
        }

        @Override
        public Tensor inverse(Tensor y) {
            return y.clone();
        }

        @Override
        public Tensor logAbsDetJacobian(Tensor x, Tensor y) {
            return torch.zeros_like(x);
        }
    }

    /**
     * Y = exp(X). Jacobian: dY/dX = exp(X) = Y ⇒ log|J| = X.
     */
    public static final class ExpTransform extends DistributionTransform {
        @Override public int eventDim() { return 0; }

        @Override
        public Tensor forward(Tensor x) {
            return torch.exp(x);
        }

        @Override
        public Tensor inverse(Tensor y) {
            Tensor safe = torch.clamp(
                    y,
                    new ScalarOptional(new Scalar(1e-12)),
                    new ScalarOptional(new Scalar(1e30))
            );
            Tensor out = torch.log(safe);
            safe.close();
            return out;
        }

        @Override
        public Tensor logAbsDetJacobian(Tensor x, Tensor y) {
            // log|exp(x)| = x
            return x.clone();
        }
    }

    /**
     * Y = loc + scale * X. Jacobian: scale (elementwise) ⇒ log|J| = log|scale|.
     */
    public static final class AffineTransform extends DistributionTransform {
        private final Tensor loc;
        private final Tensor scale;

        public AffineTransform(Tensor loc, Tensor scale) {
            if (loc == null || scale == null) {
                throw new IllegalArgumentException("AffineTransform loc/scale cannot be null");
            }
            this.loc = loc.clone();
            this.scale = scale.clone();
        }

        @Override public int eventDim() { return 0; }

        public Tensor getLoc() { return loc; }
        public Tensor getScale() { return scale; }

        @Override
        public Tensor forward(Tensor x) {
            return loc.add(scale.mul(x));
        }

        @Override
        public Tensor inverse(Tensor y) {
            return y.sub(loc).div(scale);
        }

        @Override
        public Tensor logAbsDetJacobian(Tensor x, Tensor y) {
            Tensor absScale = torch.abs(scale);
            Tensor logScale = torch.log(absScale);
            Tensor out = logScale.expand_as(x).clone();
            absScale.close();
            logScale.close();
            return out;
        }

        @Override
        public void close() {
            loc.close();
            scale.close();
        }
    }

    /**
     * Y = sigmoid(X) = 1/(1+e^{-X}).
     * Inverse: logit(Y) = log(Y) - log(1-Y).
     * log|J| = -softplus(-X) - softplus(X).
     */
    public static final class SigmoidTransform extends DistributionTransform {
        private static final float EPS = 1e-7f;

        @Override public int eventDim() { return 0; }

        @Override
        public Tensor forward(Tensor x) {
            return x.sigmoid();
        }

        @Override
        public Tensor inverse(Tensor y) {
            Tensor safe = torch.clamp(
                    y,
                    new ScalarOptional(new Scalar(EPS)),
                    new ScalarOptional(new Scalar(1.0f - EPS))
            );
            Tensor oneMinus = torch.sub(torch.ones_like(safe), safe);
            Tensor out = torch.log(safe).sub(torch.log(oneMinus));
            safe.close();
            oneMinus.close();
            return out;
        }

        @Override
        public Tensor logAbsDetJacobian(Tensor x, Tensor y) {
            // stable: -softplus(-x) - softplus(x)
            Tensor sp1 = torch.softplus(x.neg());
            Tensor sp2 = torch.softplus(x);
            Tensor out = sp1.neg().sub(sp2);
            sp1.close();
            sp2.close();
            return out;
        }
    }

    /**
     * Softplus Y = log(1+exp(X)).
     * Inverse: X = log(exp(Y)-1) = Y + log1p(-exp(-Y)).
     * log|J| = log(sigmoid(X)) = -softplus(-X).
     */
    public static final class SoftplusTransform extends DistributionTransform {
        @Override public int eventDim() { return 0; }

        @Override
        public Tensor forward(Tensor x) {
            return torch.softplus(x);
        }

        @Override
        public Tensor inverse(Tensor y) {
            // log(exp(y)-1) = y + log1p(-exp(-y)) for y > 0
            Tensor negExp = torch.exp(y.neg());
            Tensor out = y.add(torch.log1p(negExp.neg()));
            negExp.close();
            return out;
        }

        @Override
        public Tensor logAbsDetJacobian(Tensor x, Tensor y) {
            Tensor sp = torch.softplus(x.neg());
            Tensor out = sp.neg();
            sp.close();
            return out;
        }
    }

    /**
     * Abs transform Y = |X|. Not bijective; inverse maps to +Y (positive branch).
     * log|J| = 0 on the positive support (HalfNormal usage).
     */
    public static final class AbsTransform extends DistributionTransform {
        @Override public int eventDim() { return 0; }

        @Override
        public boolean bijective() { return false; }

        @Override
        public Tensor forward(Tensor x) {
            return torch.abs(x);
        }

        @Override
        public Tensor inverse(Tensor y) {
            return y.clone();
        }

        @Override
        public Tensor logAbsDetJacobian(Tensor x, Tensor y) {
            return torch.zeros_like(x);
        }
    }

    /**
     * Tanh squash Y = tanh(X). Used heavily in SAC.
     * Inverse: atanh(Y) = 0.5 * log((1+y)/(1-y)).
     * log|J| = log(1 - tanh²(x)) = 2*(log2 - x - softplus(-2x))  (stable).
     */
    public static final class TanhTransform extends DistributionTransform {
        private static final float EPS = 1e-6f;

        @Override public int eventDim() { return 0; }

        @Override
        public Tensor forward(Tensor x) {
            return x.tanh();
        }

        @Override
        public Tensor inverse(Tensor y) {
            Tensor safe = torch.clamp(
                    y,
                    new ScalarOptional(new Scalar(-1.0f + EPS)),
                    new ScalarOptional(new Scalar(1.0f - EPS))
            );
            // 0.5 * log((1+y)/(1-y))
            Tensor num = torch.add(safe, new Scalar(1.0f));
            Tensor den = torch.sub(torch.ones_like(safe), safe);
            Tensor out = torch.log(num.div(den)).mul(new Scalar(0.5f));
            safe.close();
            num.close();
            den.close();
            return out;
        }

        @Override
        public Tensor logAbsDetJacobian(Tensor x, Tensor y) {
            // stable form: 2*(log(2) - x - softplus(-2x))
            Tensor twoX = x.mul(new Scalar(2.0f));
            Tensor sp = torch.softplus(twoX.neg());
            Tensor log2 = torch.tensor(Math.log(2.0), x.options());
            Tensor out = log2.sub(x).sub(sp).mul(new Scalar(2.0f));
            twoX.close();
            sp.close();
            log2.close();
            return out;
        }
    }

    /**
     * Composition T = T_{n-1} ∘ … ∘ T_0  (applied left-to-right on samples).
     * log|J_T| = Σ_i log|J_{T_i}| along intermediate points.
     */
    public static final class ComposeTransform extends DistributionTransform {
        private final DistributionTransform[] parts;

        public ComposeTransform(DistributionTransform... transforms) {
            if (transforms == null || transforms.length == 0) {
                throw new IllegalArgumentException("ComposeTransform requires ≥1 transform");
            }
            this.parts = transforms.clone();
        }

        public DistributionTransform[] getParts() {
            return parts.clone();
        }

        @Override
        public int eventDim() {
            int max = 0;
            for (DistributionTransform t : parts) {
                max = Math.max(max, t.eventDim());
            }
            return max;
        }

        @Override
        public boolean bijective() {
            for (DistributionTransform t : parts) {
                if (!t.bijective()) return false;
            }
            return true;
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor cur = x;
            for (int i = 0; i < parts.length; i++) {
                Tensor next = parts[i].forward(cur);
                if (cur != x) {
                    cur.close();
                }
                cur = next;
            }
            return cur;
        }

        @Override
        public Tensor inverse(Tensor y) {
            Tensor cur = y;
            for (int i = parts.length - 1; i >= 0; i--) {
                Tensor next = parts[i].inverse(cur);
                if (cur != y) {
                    cur.close();
                }
                cur = next;
            }
            return cur;
        }

        @Override
        public Tensor logAbsDetJacobian(Tensor x, Tensor y) {
            // recompute intermediates: x0=x, x1=T0(x0), ..., xn
            Tensor[] xs = new Tensor[parts.length + 1];
            xs[0] = x;
            for (int i = 0; i < parts.length; i++) {
                xs[i + 1] = parts[i].forward(xs[i]);
            }
            Tensor total = null;
            for (int i = 0; i < parts.length; i++) {
                Tensor j = parts[i].logAbsDetJacobian(xs[i], xs[i + 1]);
                if (total == null) {
                    total = j;
                } else {
                    Tensor sum = total.add(j);
                    total.close();
                    j.close();
                    total = sum;
                }
            }
            // free intermediates (not original x)
            for (int i = 1; i < xs.length; i++) {
                xs[i].close();
            }
            return total != null ? total : torch.zeros_like(x);
        }

        @Override
        public void close() {
            for (DistributionTransform t : parts) {
                try { t.close(); } catch (Exception ignored) {}
            }
        }
    }
}
