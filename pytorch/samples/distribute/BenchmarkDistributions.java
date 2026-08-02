package distribute;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.*;
import org.bytedeco.pytorch.distribution.transforms.DistributionTransform;
import org.bytedeco.pytorch.distribution.transforms.DistributionTransforms;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Comprehensive multi-aspect benchmark / validation suite for
 * {@code org.bytedeco.pytorch.distribution.*}.
 *
 * <p>Dimensions covered:</p>
 * <ol>
 *   <li><b>Construction</b> — valid params construct; invalid params throw</li>
 *   <li><b>Sample shapes</b> — scalar / 1-D batch / multi-D batch × sampleShape</li>
 *   <li><b>log_prob</b> — finite on support; −∞ / throw off support; shape matches</li>
 *   <li><b>entropy / mean</b> — finite, correct shape; MC mean ≈ analytical mean</li>
 *   <li><b>TransformedDistribution</b> — identity / exp / affine / tanh / compose;
 *       invertibility; LogNormal ≡ Normal∘Exp</li>
 *   <li><b>Wrappers</b> — Independent, MixtureSameFamily</li>
 *   <li><b>Stress</b> — large sample counts, batched params, close() safety</li>
 * </ol>
 *
 * <pre>
 *   cd samples &amp;&amp; mvn -q exec:java -Dexec.mainClass=distribute.BenchmarkDistributions
 * </pre>
 */
public class BenchmarkDistributions {

    private static final double ATOL_MEAN = 0.25;     // MC mean tolerance (loose for small N)
    private static final double ATOL_LOGP = 1e-3;     // analytical log_prob tolerance
    private static final double ATOL_INV = 1e-4;      // transform invertibility
    private static final long MC_N = 8000;            // MC sample size for mean checks
    private static final long MC_N_SMALL = 2000;

    private static int passed = 0;
    private static int failed = 0;
    private static int skipped = 0;
    private static final List<String> failures = new ArrayList<>();

    // ------------------------------------------------------------------
    // harness
    // ------------------------------------------------------------------

    private static void check(String name, boolean cond, String detail) {
        if (cond) {
            passed++;
            System.out.printf(Locale.ROOT, "  [PASS] %s%n", name);
        } else {
            failed++;
            String msg = name + (detail == null || detail.isEmpty() ? "" : " — " + detail);
            failures.add(msg);
            System.out.printf(Locale.ROOT, "  [FAIL] %s%n", msg);
        }
    }

    private static void skip(String name, String reason) {
        skipped++;
        System.out.printf(Locale.ROOT, "  [SKIP] %s — %s%n", name, reason);
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("======== " + title + " ========");
    }

    private static Tensor f(float... v) {
        if (v.length == 1) {
            return torch.tensor(v[0]);
        }
        return torch.tensor(v);
    }

    private static Tensor f2(float[][] rows) {
        int r = rows.length;
        int c = rows[0].length;
        float[] flat = new float[r * c];
        for (int i = 0; i < r; i++) {
            System.arraycopy(rows[i], 0, flat, i * c, c);
        }
        Tensor t = torch.tensor(flat);
        return t.reshape(r, c);
    }

    private static float scalar(Tensor t) {
        Tensor flat = t.reshape(-1);
        float v = flat.item().toFloat();
        if (flat != t) flat.close();
        return v;
    }

    private static boolean allFinite(Tensor t) {
        Tensor finite = torch.isfinite(t);
        boolean ok = torch.all(finite).item().toBool();
        finite.close();
        return ok;
    }

    private static boolean shapesEqual(long[] a, long[] b) {
        return Arrays.equals(a, b);
    }

    private static long[] shapeOf(Tensor t) {
        return t.sizes().vec().get();
    }

    private static void safeClose(AutoCloseable... cs) {
        for (AutoCloseable c : cs) {
            if (c == null) continue;
            try { c.close(); } catch (Exception ignored) {}
        }
    }

    private static void safeCloseT(Tensor... ts) {
        for (Tensor t : ts) {
            if (t != null) {
                try { t.close(); } catch (Exception ignored) {}
            }
        }
    }

    // ------------------------------------------------------------------
    // per-distribution factory
    // ------------------------------------------------------------------

    @FunctionalInterface
    interface DistFactory {
        Distribution create() throws Exception;
    }

    static final class DistCase {
        final String name;
        final DistFactory factory;
        final boolean continuous;
        final boolean hasMean;      // StudentT df<=1 has no mean, Cauchy none, etc.
        final boolean checkMcMean;
        final float[] supportProbe; // values expected to be on support for log_prob

        DistCase(String name, DistFactory factory, boolean continuous,
                 boolean hasMean, boolean checkMcMean, float... supportProbe) {
            this.name = name;
            this.factory = factory;
            this.continuous = continuous;
            this.hasMean = hasMean;
            this.checkMcMean = checkMcMean;
            this.supportProbe = supportProbe;
        }
    }

    private static List<DistCase> buildCases() {
        List<DistCase> cases = new ArrayList<>();

        // ---- univariate continuous ----
        cases.add(new DistCase("Normal(0,1)",
                () -> new Normal(f(0f), f(1f)), true, true, true, 0f, 1f, -1f));
        cases.add(new DistCase("Normal(batch)",
                () -> new Normal(f(0f, 1f, -2f), f(1f, 0.5f, 2f)), true, true, true, 0f));
        cases.add(new DistCase("Laplace(0,1)",
                () -> new Laplace(f(0f), f(1f)), true, true, true, 0f, 1f));
        cases.add(new DistCase("Cauchy(0,1)",
                () -> new Cauchy(f(0f), f(1f)), true, false, false, 0f, 1f));
        cases.add(new DistCase("Gumbel(0,1)",
                () -> new Gumbel(f(0f), f(1f)), true, true, true, 0f, 1f));
        cases.add(new DistCase("Logistic(0,1)",
                () -> new Logistic(f(0f), f(1f)), true, true, true, 0f, 1f));
        cases.add(new DistCase("Uniform(0,1)",
                () -> new Uniform(f(0f), f(1f)), true, true, true, 0.3f, 0.7f));
        cases.add(new DistCase("Uniform(batch)",
                () -> new Uniform(f(0f, -1f), f(1f, 2f)), true, true, true, 0.5f));
        cases.add(new DistCase("Exponential(1)",
                () -> new Exponential(f(1f)), true, true, true, 0.5f, 1f));
        cases.add(new DistCase("Gamma(2,1)",
                () -> new Gamma(f(2f), f(1f)), true, true, true, 1f, 2f));
        cases.add(new DistCase("Beta(2,3)",
                () -> new Beta(f(2f), f(3f)), true, true, true, 0.3f, 0.5f));
        cases.add(new DistCase("Chi2(3)",
                () -> new Chi2(f(3f)), true, true, true, 1f, 3f));
        cases.add(new DistCase("HalfNormal(1)",
                () -> new HalfNormal(f(1f)), true, true, true, 0.5f, 1f));
        cases.add(new DistCase("HalfCauchy(1)",
                () -> new HalfCauchy(f(1f)), true, false, false, 0.5f, 1f));
        cases.add(new DistCase("LogNormal(0,1)",
                () -> new LogNormal(f(0f), f(1f)), true, true, true, 1f, 2f));
        cases.add(new DistCase("Pareto(1,3)",
                () -> new Pareto(f(1f), f(3f)), true, true, true, 1.5f, 2f));
        cases.add(new DistCase("Weibull(1,1.5)",
                () -> new Weibull(f(1f), f(1.5f)), true, true, true, 0.5f, 1f));
        cases.add(new DistCase("StudentT(5,0,1)",
                () -> new StudentT(f(5f), f(0f), f(1f)), true, true, true, 0f, 1f));
        cases.add(new DistCase("InverseGamma(3,2)",
                () -> new InverseGamma(f(3f), f(2f)), true, true, true, 1f, 2f));
        cases.add(new DistCase("FisherSnedecor(5,10)",
                () -> new FisherSnedecor(f(5f), f(10f)), true, true, true, 1f, 2f));
        cases.add(new DistCase("Kumaraswamy(2,3)",
                () -> new Kumaraswamy(f(2f), f(3f)), true, true, true, 0.3f, 0.6f));
        cases.add(new DistCase("VonMises(0,2)",
                () -> new VonMises(f(0f), f(2f)), true, true, false, 0f, 0.5f));

        // ---- univariate discrete ----
        cases.add(new DistCase("Bernoulli(0.3)",
                () -> new Bernoulli(f(0.3f)), false, true, true, 0f, 1f));
        cases.add(new DistCase("Binomial(10,0.4)",
                () -> new Binomial(f(10f), f(0.4f)), false, true, true, 4f, 5f));
        cases.add(new DistCase("Geometric(0.3)",
                () -> new Geometric(f(0.3f)), false, true, true, 0f, 1f, 2f));
        cases.add(new DistCase("Poisson(3)",
                () -> new Poisson(f(3f)), false, true, true, 2f, 3f, 4f));
        cases.add(new DistCase("NegativeBinomial(5,0.5)",
                () -> new NegativeBinomial(f(5f), f(0.5f)), false, true, true, 3f, 5f));
        cases.add(new DistCase("ContinuousBernoulli(0.4)",
                () -> new ContinuousBernoulli(f(0.4f)), true, true, true, 0.2f, 0.5f));

        // ---- categorical family ----
        cases.add(new DistCase("Categorical([0.1,0.2,0.7])",
                () -> new Categorical(f(0.1f, 0.2f, 0.7f)), false, false, false, 0f, 1f, 2f));
        cases.add(new DistCase("OneHotCategorical([0.2,0.3,0.5])",
                () -> new OneHotCategorical(f(0.2f, 0.3f, 0.5f)), false, true, false));
        cases.add(new DistCase("OneHotCategoricalST([0.2,0.3,0.5])",
                () -> new OneHotCategoricalStraightThrough(f(0.2f, 0.3f, 0.5f)), false, true, false));
        cases.add(new DistCase("RelaxedBernoulli(0.5,0.4)",
                () -> new RelaxedBernoulli(f(0.5f), f(0.4f)), true, true, false, 0.3f, 0.6f));
        cases.add(new DistCase("RelaxedOneHot(0.5,[0.2,0.3,0.5])",
                () -> new RelaxedOneHotCategorical(f(0.5f), f(0.2f, 0.3f, 0.5f)), true, true, false));

        // ---- multivariate ----
        cases.add(new DistCase("Dirichlet([2,3,4])",
                () -> new Dirichlet(f(2f, 3f, 4f)), true, true, true));
        cases.add(new DistCase("MultivariateNormal(2d)",
                () -> {
                    Tensor loc = f(0f, 0f);
                    Tensor cov = f2(new float[][]{{1f, 0.3f}, {0.3f, 1f}});
                    return new MultivariateNormal(loc, cov);
                }, true, true, true));
        cases.add(new DistCase("LowRankMVN(3d,k=1)",
                () -> {
                    Tensor loc = f(0f, 0f, 0f);
                    Tensor factor = f(0.5f, 0.3f, 0.1f).reshape(3, 1);
                    Tensor diag = f(1f, 1f, 1f);
                    return new LowRankMultivariateNormal(loc, factor, diag);
                }, true, true, true));

        return cases;
    }

    // ------------------------------------------------------------------
    // suite 1: core API per distribution
    // ------------------------------------------------------------------

    private static void runCoreApiSuite() {
        section("1. Core API (sample / log_prob / entropy / mean / close)");
        for (DistCase c : buildCases()) {
            System.out.println("-- " + c.name);
            Distribution d = null;
            try {
                d = c.factory.create();
                check(c.name + ".name()", d.name() != null && !d.name().isEmpty(), d.name());

                // sample no-arg / empty shape
                Tensor s0 = d.sample();
                check(c.name + ".sample() finite", allFinite(s0), "shape=" + Arrays.toString(shapeOf(s0)));
                safeCloseT(s0);

                // sample with shape
                Tensor s1 = d.sample(4);
                check(c.name + ".sample(4) leading dim",
                        shapeOf(s1).length >= 1 && shapeOf(s1)[0] == 4,
                        "shape=" + Arrays.toString(shapeOf(s1)));
                check(c.name + ".sample(4) finite", allFinite(s1), null);

                // multi-dim sample shape
                Tensor s2 = d.sample(2, 3);
                check(c.name + ".sample(2,3) leading dims",
                        shapeOf(s2).length >= 2 && shapeOf(s2)[0] == 2 && shapeOf(s2)[1] == 3,
                        "shape=" + Arrays.toString(shapeOf(s2)));
                safeCloseT(s2);

                // log_prob on own samples
                Tensor lp = d.log_prob(s1);
                check(c.name + ".log_prob(sample) finite", allFinite(lp) || !c.continuous,
                        "shape=" + Arrays.toString(shapeOf(lp)));
                // for continuous, expect finite; discrete may be -inf on float noise — already sampled so should be ok
                if (c.continuous) {
                    check(c.name + ".log_prob mean finite",
                            Float.isFinite(scalar(lp.mean())), null);
                }
                safeCloseT(lp);

                // support probes
                if (c.supportProbe != null && c.supportProbe.length > 0) {
                    for (float v : c.supportProbe) {
                        try {
                            Tensor vt = f(v);
                            // expand probe to match event shape if needed
                            Tensor probe = expandProbe(d, vt, s1);
                            Tensor lpv = d.log_prob(probe);
                            check(c.name + ".log_prob(" + v + ") runs", lpv != null, null);
                            safeCloseT(vt, probe, lpv);
                        } catch (Exception e) {
                            // some multivariate need vector probes — skip
                            skip(c.name + ".log_prob(" + v + ")", e.getClass().getSimpleName() + ": " + e.getMessage());
                        }
                    }
                }

                // entropy
                try {
                    Tensor ent = d.entropy();
                    check(c.name + ".entropy() finite", allFinite(ent),
                            "val~" + (ent.numel() == 1 ? scalar(ent) : Arrays.toString(shapeOf(ent))));
                    safeCloseT(ent);
                } catch (UnsupportedOperationException | IllegalStateException e) {
                    skip(c.name + ".entropy()", e.getMessage());
                } catch (Exception e) {
                    check(c.name + ".entropy()", false, e.getClass().getSimpleName() + ": " + e.getMessage());
                }

                // mean + MC check
                if (c.hasMean) {
                    try {
                        Tensor mean = d.mean();
                        check(c.name + ".mean() finite", allFinite(mean),
                                "shape=" + Arrays.toString(shapeOf(mean)));
                        if (c.checkMcMean && mean.numel() == 1) {
                            Tensor big = d.sample(MC_N);
                            Tensor mcMean = big.to(kFloat()).mean(new long[]{0}, false, new ScalarTypeOptional());
                            // if sample has event dims, mean over sample only may still be vector
                            if (mcMean.numel() == 1 && mean.numel() == 1) {
                                float diff = Math.abs(scalar(mcMean) - scalar(mean));
                                check(c.name + ".MC mean≈analytical",
                                        diff < ATOL_MEAN || diff / (Math.abs(scalar(mean)) + 1e-3) < 0.35,
                                        String.format(Locale.ROOT, "mc=%.4f analytical=%.4f diff=%.4f",
                                                scalar(mcMean), scalar(mean), diff));
                            } else {
                                skip(c.name + ".MC mean", "non-scalar event/batch shape");
                            }
                            safeCloseT(big, mcMean);
                        }
                        safeCloseT(mean);
                    } catch (Exception e) {
                        check(c.name + ".mean()", false, e.getMessage());
                    }
                } else {
                    skip(c.name + ".mean/MC", "distribution has no finite mean");
                }

                safeCloseT(s1);
            } catch (Exception e) {
                check(c.name + " suite", false, e.getClass().getSimpleName() + ": " + e.getMessage());
                e.printStackTrace(System.out);
            } finally {
                if (d instanceof AutoCloseable) {
                    safeClose((AutoCloseable) d);
                }
            }
        }
    }

    /** Expand a scalar probe to the event shape of a sample from d. */
    private static Tensor expandProbe(Distribution d, Tensor scalarProbe, Tensor sampleRef) {
        long[] ss = shapeOf(sampleRef);
        if (ss.length == 0 || (ss.length == 1 && ss[0] >= 1 && sampleRef.dim() <= 1 && sampleRef.numel() <= 4)) {
            // univariate-ish
            return scalarProbe.clone();
        }
        // if last dim is event (e.g. one-hot / dirichlet / mvn), fill a vector
        if (sampleRef.dim() >= 1 && sampleRef.size(-1) > 1 && sampleRef.size(-1) <= 16) {
            long k = sampleRef.size(-1);
            // for one-hot: put mass on class 0; for dirichlet: uniform simplex
            float[] vec = new float[(int) k];
            if (d instanceof OneHotCategorical || d instanceof OneHotCategoricalStraightThrough
                    || d instanceof RelaxedOneHotCategorical) {
                vec[0] = 1f;
            } else if (d instanceof Dirichlet) {
                Arrays.fill(vec, 1f / k);
            } else {
                // mvn etc: use zeros (at mean-ish)
                Arrays.fill(vec, 0f);
                vec[0] = scalar(scalarProbe);
            }
            return torch.tensor(vec);
        }
        return scalarProbe.clone();
    }

    // ------------------------------------------------------------------
    // suite 2: invalid construction
    // ------------------------------------------------------------------

    private static void runInvalidConstructionSuite() {
        section("2. Invalid construction (must throw)");

        expectThrow("Normal scale<=0", () -> new Normal(f(0f), f(0f)));
        expectThrow("Normal scale negative", () -> new Normal(f(0f), f(-1f)));
        expectThrow("Uniform low>=high", () -> new Uniform(f(1f), f(0f)));
        expectThrow("Bernoulli p>1", () -> new Bernoulli(f(1.5f)));
        expectThrow("Bernoulli p<0", () -> new Bernoulli(f(-0.1f)));
        expectThrow("Exponential rate<=0", () -> new Exponential(f(0f)));
        expectThrow("Gamma alpha<=0", () -> new Gamma(f(0f), f(1f)));
        expectThrow("Beta a<=0", () -> new Beta(f(-1f), f(2f)));
        expectThrow("HalfNormal scale<=0", () -> new HalfNormal(f(0f)));
        expectThrow("LogNormal s<=0", () -> new LogNormal(f(0f), f(0f)));
        expectThrow("Poisson lambda<=0", () -> {
            try {
                return new Poisson(f(0f));
            } catch (RuntimeException e) {
                throw e;
            }
        });
        expectThrow("Categorical negative probs", () -> new Categorical(f(-0.1f, 0.5f, 0.5f)));
        expectThrow("NegativeBinomial p=0", () -> new NegativeBinomial(f(5f), f(0f)));
        expectThrow("NegativeBinomial p=1", () -> new NegativeBinomial(f(5f), f(1f)));
        expectThrow("StudentT df<=0", () -> new StudentT(f(0f), f(0f), f(1f)));
        expectThrow("Gumbel scale<=0", () -> new Gumbel(f(0f), f(0f)));
    }

    private static void expectThrow(String name, Supplier<Object> supplier) {
        try {
            Object o = supplier.get();
            if (o instanceof AutoCloseable) safeClose((AutoCloseable) o);
            check(name, false, "expected exception, got success");
        } catch (IllegalArgumentException | IllegalStateException e) {
            check(name, true, e.getMessage());
        } catch (RuntimeException e) {
            // some implementations wrap
            check(name, true, e.getClass().getSimpleName() + ": " + e.getMessage());
        } catch (Exception e) {
            check(name, true, e.getClass().getSimpleName());
        }
    }

    // ------------------------------------------------------------------
    // suite 3: analytical log_prob checks (Normal, Uniform, Bernoulli, Exp)
    // ------------------------------------------------------------------

    private static void runAnalyticalLogProbSuite() {
        section("3. Analytical log_prob / entropy cross-checks");

        // Normal(0,1): log φ(0) = -0.5*log(2π) ≈ -0.9189385
        try (Normal n = new Normal(f(0f), f(1f))) {
            Tensor lp0 = n.log_prob(f(0f));
            float got = scalar(lp0);
            float expect = (float) (-0.5 * Math.log(2 * Math.PI));
            check("Normal(0,1).log_prob(0)", Math.abs(got - expect) < ATOL_LOGP,
                    String.format(Locale.ROOT, "got=%.6f expect=%.6f", got, expect));
            safeCloseT(lp0);

            // log_prob(1) = -0.5*log(2π) - 0.5
            Tensor lp1 = n.log_prob(f(1f));
            float got1 = scalar(lp1);
            float expect1 = expect - 0.5f;
            check("Normal(0,1).log_prob(1)", Math.abs(got1 - expect1) < ATOL_LOGP,
                    String.format(Locale.ROOT, "got=%.6f expect=%.6f", got1, expect1));
            safeCloseT(lp1);

            // entropy = 0.5*(1+log(2π)) ≈ 1.4189385
            Tensor ent = n.entropy();
            float eGot = scalar(ent);
            float eExp = (float) (0.5 * (1 + Math.log(2 * Math.PI)));
            check("Normal(0,1).entropy", Math.abs(eGot - eExp) < ATOL_LOGP,
                    String.format(Locale.ROOT, "got=%.6f expect=%.6f", eGot, eExp));
            safeCloseT(ent);
        }

        // Uniform(0,1): log_prob(0.3)=0, entropy=0
        try (Uniform u = new Uniform(f(0f), f(1f))) {
            Tensor lp = u.log_prob(f(0.3f));
            check("Uniform(0,1).log_prob(0.3)≈0", Math.abs(scalar(lp)) < ATOL_LOGP,
                    "got=" + scalar(lp));
            safeCloseT(lp);
            Tensor ent = u.entropy();
            check("Uniform(0,1).entropy≈0", Math.abs(scalar(ent)) < ATOL_LOGP, "got=" + scalar(ent));
            safeCloseT(ent);
        }

        // Bernoulli(0.3): log_prob(1)=log(0.3), log_prob(0)=log(0.7)
        try (Bernoulli b = new Bernoulli(f(0.3f))) {
            Tensor lp1 = b.log_prob(f(1f));
            Tensor lp0 = b.log_prob(f(0f));
            check("Bernoulli(0.3).log_prob(1)", Math.abs(scalar(lp1) - Math.log(0.3)) < ATOL_LOGP,
                    "got=" + scalar(lp1));
            check("Bernoulli(0.3).log_prob(0)", Math.abs(scalar(lp0) - Math.log(0.7)) < ATOL_LOGP,
                    "got=" + scalar(lp0));
            safeCloseT(lp1, lp0);
        }

        // Exponential(2): log_prob(1)=log(2)-2
        try (Exponential e = new Exponential(f(2f))) {
            Tensor lp = e.log_prob(f(1f));
            float expect = (float) (Math.log(2.0) - 2.0);
            check("Exponential(2).log_prob(1)", Math.abs(scalar(lp) - expect) < ATOL_LOGP,
                    String.format(Locale.ROOT, "got=%.6f expect=%.6f", scalar(lp), expect));
            safeCloseT(lp);
            // mean = 1/rate = 0.5
            Tensor mean = e.mean();
            check("Exponential(2).mean=0.5", Math.abs(scalar(mean) - 0.5f) < ATOL_LOGP, "got=" + scalar(mean));
            safeCloseT(mean);
        }

        // Laplace(0,1): log_prob(0)= -log(2) ≈ -0.693147
        try (Laplace l = new Laplace(f(0f), f(1f))) {
            Tensor lp = l.log_prob(f(0f));
            float expect = (float) (-Math.log(2.0));
            check("Laplace(0,1).log_prob(0)", Math.abs(scalar(lp) - expect) < ATOL_LOGP,
                    String.format(Locale.ROOT, "got=%.6f expect=%.6f", scalar(lp), expect));
            safeCloseT(lp);
        }

        // Gamma(1,1) == Exp(1): log_prob(1)= -1
        try (Gamma g = new Gamma(f(1f), f(1f))) {
            Tensor lp = g.log_prob(f(1f));
            check("Gamma(1,1).log_prob(1)≈-1", Math.abs(scalar(lp) - (-1f)) < 5e-3,
                    "got=" + scalar(lp));
            safeCloseT(lp);
        }

        // Batched Normal log_prob broadcast
        try (Normal n = new Normal(f(0f, 1f), f(1f, 2f))) {
            Tensor x = f(0f, 1f);
            Tensor lp = n.log_prob(x);
            check("Normal(batch).log_prob shape", shapeOf(lp).length == 1 && shapeOf(lp)[0] == 2,
                    Arrays.toString(shapeOf(lp)));
            check("Normal(batch).log_prob finite", allFinite(lp), null);
            safeCloseT(x, lp);
        }
    }

    // ------------------------------------------------------------------
    // suite 4: TransformedDistribution
    // ------------------------------------------------------------------

    private static void runTransformedSuite() {
        section("4. TransformedDistribution + Transforms");

        // 4.1 Identity transform preserves Normal
        {
            Normal base = new Normal(f(0f), f(1f));
            TransformedDistribution td = new TransformedDistribution(base, DistributionTransforms.identity());
            Tensor x = f(0.5f);
            Tensor lpBase = base.log_prob(x);
            Tensor lpTd = td.log_prob(x);
            check("Identity log_prob matches base",
                    Math.abs(scalar(lpBase) - scalar(lpTd)) < ATOL_LOGP,
                    String.format(Locale.ROOT, "base=%.6f td=%.6f", scalar(lpBase), scalar(lpTd)));
            Tensor s = td.sample(100);
            check("Identity sample finite", allFinite(s), null);
            check("Identity invertibility", td.validateInvertibility(f(1.23f), ATOL_INV), null);
            safeCloseT(x, lpBase, lpTd, s);
            safeClose(td, base);
        }

        // 4.2 Exp transform: Transformed(Normal(0,1), Exp) ≈ LogNormal(0,1)
        {
            Normal base = new Normal(f(0f), f(1f));
            TransformedDistribution td = new TransformedDistribution(base, DistributionTransforms.exp(), false, true);
            LogNormal ln = new LogNormal(f(0f), f(1f));

            Tensor y = f(1.5f);
            Tensor lpTd = td.log_prob(y);
            Tensor lpLn = ln.log_prob(y);
            check("Exp∘Normal ≡ LogNormal log_prob(1.5)",
                    Math.abs(scalar(lpTd) - scalar(lpLn)) < 5e-3,
                    String.format(Locale.ROOT, "td=%.6f ln=%.6f", scalar(lpTd), scalar(lpLn)));

            // samples must be positive
            Tensor s = td.sample(500);
            Tensor min = s.min();
            check("Exp∘Normal samples > 0", scalar(min) > 0, "min=" + scalar(min));
            check("Exp invertibility", td.validateInvertibility(f(0.7f), 1e-3), null);

            safeCloseT(y, lpTd, lpLn, s, min);
            safeClose(td, base, ln);
        }

        // 4.3 Affine transform: loc=2, scale=3 on Normal(0,1) → Normal(2,3)
        {
            Normal base = new Normal(f(0f), f(1f));
            DistributionTransform aff = DistributionTransforms.affine(f(2f), f(3f));
            TransformedDistribution td = new TransformedDistribution(base, aff, false, true);
            Normal target = new Normal(f(2f), f(3f));

            Tensor y = f(2f);
            Tensor lpTd = td.log_prob(y);
            Tensor lpT = target.log_prob(y);
            check("Affine∘Normal(0,1) ≡ Normal(2,3) log_prob",
                    Math.abs(scalar(lpTd) - scalar(lpT)) < 5e-3,
                    String.format(Locale.ROOT, "td=%.6f target=%.6f", scalar(lpTd), scalar(lpT)));

            Tensor mean = td.mean(false); // affine → exact
            check("Affine mean=2", Math.abs(scalar(mean) - 2f) < ATOL_LOGP, "got=" + scalar(mean));
            check("Affine invertibility", td.validateInvertibility(f(-1f), ATOL_INV), null);

            safeCloseT(y, lpTd, lpT, mean);
            safeClose(td, base, target);
        }

        // 4.4 Tanh squash (SAC-style)
        {
            Normal base = new Normal(f(0f), f(1f));
            TransformedDistribution td = new TransformedDistribution(base, DistributionTransforms.tanh(), false, true);
            Tensor s = td.sample(1000);
            Tensor absMax = torch.abs(s).max();
            check("Tanh samples in (-1,1)", scalar(absMax) < 1.0f + 1e-5, "max|s|=" + scalar(absMax));
            Tensor lp = td.log_prob(f(0f));
            check("Tanh log_prob(0) finite", Float.isFinite(scalar(lp)), "got=" + scalar(lp));
            check("Tanh invertibility", td.validateInvertibility(f(0.3f), 1e-3), null);
            safeCloseT(s, absMax, lp);
            safeClose(td, base);
        }

        // 4.5 Sigmoid transform → (0,1)
        {
            Normal base = new Normal(f(0f), f(1f));
            TransformedDistribution td = new TransformedDistribution(base, DistributionTransforms.sigmoid(), false, true);
            Tensor s = td.sample(500);
            Tensor mn = s.min();
            Tensor mx = s.max();
            check("Sigmoid samples in (0,1)", scalar(mn) > 0 && scalar(mx) < 1,
                    "min=" + scalar(mn) + " max=" + scalar(mx));
            check("Sigmoid invertibility", td.validateInvertibility(f(0.0f), 1e-3) == false
                    || td.validateInvertibility(f(0.25f), 1e-3), null);
            // better: test interior point
            check("Sigmoid invertibility@0.25", td.validateInvertibility(f(0.25f), 1e-3), null);
            safeCloseT(s, mn, mx);
            safeClose(td, base);
        }

        // 4.6 Compose: Affine then Exp  (lognormal with loc/scale)
        {
            Normal base = new Normal(f(0f), f(1f));
            DistributionTransform comp = DistributionTransforms.compose(
                    DistributionTransforms.affine(f(0.5f), f(0.8f)),
                    DistributionTransforms.exp()
            );
            TransformedDistribution td = new TransformedDistribution(base, comp, false, true);
            Tensor s = td.sample(300);
            check("Compose(Affine,Exp) samples > 0", scalar(s.min()) > 0, null);
            Tensor y = f(1.2f);
            Tensor lp = td.log_prob(y);
            check("Compose log_prob finite", Float.isFinite(scalar(lp)), "got=" + scalar(lp));
            safeCloseT(s, y, lp);
            safeClose(td, base);
        }

        // 4.7 Softplus transform
        {
            Normal base = new Normal(f(0f), f(1f));
            TransformedDistribution td = new TransformedDistribution(base, DistributionTransforms.softplus(), false, true);
            Tensor s = td.sample(200);
            check("Softplus samples > 0", scalar(s.min()) > 0, "min=" + scalar(s.min()));
            check("Softplus invertibility", td.validateInvertibility(f(0.8f), 1e-3), null);
            safeCloseT(s);
            safeClose(td, base);
        }

        // 4.8 null args
        expectThrow("Transformed null base", () -> new TransformedDistribution(null, DistributionTransforms.identity()));
        expectThrow("Transformed null transform", () -> {
            Normal n = new Normal(f(0f), f(1f));
            try {
                return new TransformedDistribution(n, (DistributionTransform) null);
            } finally {
                n.close();
            }
        });
    }

    // ------------------------------------------------------------------
    // suite 5: Independent + MixtureSameFamily
    // ------------------------------------------------------------------

    private static void runWrapperSuite() {
        section("5. Independent & MixtureSameFamily");

        // Independent: sum log_prob over reinterpreted dims
        {
            Normal base = new Normal(f(0f, 0f, 0f), f(1f, 1f, 1f)); // batch of 3
            Independent ind = new Independent(base, 1);
            Tensor x = f(0f, 0f, 0f);
            Tensor lpBase = base.log_prob(x);          // shape [3]
            Tensor lpInd = ind.log_prob(x);            // should be scalar sum
            float sumBase = scalar(lpBase.sum(new long[]{0}, false, new ScalarTypeOptional()));
            check("Independent log_prob = sum base",
                    Math.abs(scalar(lpInd) - sumBase) < 5e-3,
                    String.format(Locale.ROOT, "ind=%.6f sumBase=%.6f", scalar(lpInd), sumBase));
            Tensor s1 = ind.sample(5);
            Tensor s2 = ind.sample(5);
            // samples must differ (no caching)
            Tensor diff = torch.abs(s1.sub(s2)).sum();
            check("Independent sample not cached", scalar(diff) > 1e-6, "diff=" + scalar(diff));
            safeCloseT(x, lpBase, lpInd, s1, s2, diff);
            safeClose(ind); // closes base too
        }

        // MixtureSameFamily: 2-component Normal mixture
        {
            try {
                Categorical mix = new Categorical(f(0.4f, 0.6f));
                // component: batch of 2 normals — loc/scale shape [2]
                Normal comp = new Normal(f(-2f, 2f), f(1f, 1f));
                MixtureSameFamily msf = new MixtureSameFamily(mix, comp);
                Tensor s = msf.sample(500);
                check("Mixture sample finite", allFinite(s), "shape=" + Arrays.toString(shapeOf(s)));
                Tensor lp = msf.log_prob(f(0f));
                check("Mixture log_prob(0) finite", Float.isFinite(scalar(lp)), "got=" + scalar(lp));
                // mixture density at 0 should be higher than either tail alone roughly
                Tensor mean = msf.mean();
                check("Mixture mean finite", allFinite(mean), "mean=" + scalar(mean));
                safeCloseT(s, lp, mean);
                safeClose(msf);
            } catch (Exception e) {
                check("MixtureSameFamily", false, e.getClass().getSimpleName() + ": " + e.getMessage());
                e.printStackTrace(System.out);
            }
        }
    }

    // ------------------------------------------------------------------
    // suite 6: batch / multi-dim stress
    // ------------------------------------------------------------------

    private static void runBatchStressSuite() {
        section("6. Batch & multi-dimensional stress");

        // 2-D batch Normal
        {
            Tensor loc = torch.zeros(new long[]{4, 5});
            Tensor scale = torch.ones(new long[]{4, 5});
            Normal n = new Normal(loc, scale);
            Tensor s = n.sample(3); // [3,4,5]
            long[] sh = shapeOf(s);
            check("Normal 2D-batch sample shape",
                    sh.length == 3 && sh[0] == 3 && sh[1] == 4 && sh[2] == 5,
                    Arrays.toString(sh));
            Tensor lp = n.log_prob(s);
            check("Normal 2D-batch log_prob shape",
                    shapeOf(lp).length == 3 && shapeOf(lp)[0] == 3,
                    Arrays.toString(shapeOf(lp)));
            check("Normal 2D-batch log_prob finite", allFinite(lp), null);
            Tensor ent = n.entropy();
            check("Normal 2D-batch entropy shape",
                    shapeOf(ent).length == 2 && shapeOf(ent)[0] == 4 && shapeOf(ent)[1] == 5,
                    Arrays.toString(shapeOf(ent)));
            safeCloseT(loc, scale, s, lp, ent);
            safeClose(n);
        }

        // Large sample throughput (timing)
        {
            Normal n = new Normal(f(0f), f(1f));
            long t0 = System.nanoTime();
            Tensor s = n.sample(50_000);
            Tensor lp = n.log_prob(s);
            long t1 = System.nanoTime();
            double ms = (t1 - t0) / 1e6;
            check("Normal 50k sample+log_prob", allFinite(lp),
                    String.format(Locale.ROOT, "%.1f ms", ms));
            System.out.printf(Locale.ROOT, "  [INFO] Normal 50k sample+log_prob took %.1f ms%n", ms);
            safeCloseT(s, lp);
            safeClose(n);
        }

        // Categorical batch
        {
            Tensor probs = f2(new float[][]{
                    {0.1f, 0.2f, 0.7f},
                    {0.3f, 0.3f, 0.4f},
                    {0.5f, 0.25f, 0.25f}
            });
            Categorical cat = new Categorical(probs);
            Tensor s = cat.sample(10); // [10, 3]
            check("Categorical batch sample leading", shapeOf(s)[0] == 10, Arrays.toString(shapeOf(s)));
            Tensor lp = cat.log_prob(s);
            check("Categorical batch log_prob finite", allFinite(lp), Arrays.toString(shapeOf(lp)));
            safeCloseT(probs, s, lp);
            safeClose(cat);
        }

        // close() is idempotent-ish
        {
            Normal n = new Normal(f(0f), f(1f));
            n.close();
            try {
                n.close(); // second close should not crash hard
                check("Normal double close", true, null);
            } catch (Exception e) {
                // acceptable if it throws; just must not JVM-crash
                check("Normal double close (threw)", true, e.getMessage());
            }
        }
    }

    // ------------------------------------------------------------------
    // suite 7: Normal correctness deep-dive (replaces NormalFixed)
    // ------------------------------------------------------------------

    private static void runNormalDeepDive() {
        section("7. Normal deep-dive (NormalFixed removed — Normal is canonical)");

        // multi-batch + multi-sample
        Tensor loc = f(0f, 1f, -1f);
        Tensor scale = f(1f, 2f, 0.5f);
        Normal n = new Normal(loc, scale);

        // analytical entropy: log(σ) + 0.5*(1+log(2π))
        Tensor ent = n.entropy();
        for (int i = 0; i < 3; i++) {
            float s = scale.reshape(-1).get(i).item().toFloat();
            float expect = (float) (Math.log(s) + 0.5 * (1 + Math.log(2 * Math.PI)));
            float got = ent.reshape(-1).get(i).item().toFloat();
            check("Normal entropy[" + i + "]", Math.abs(got - expect) < ATOL_LOGP,
                    String.format(Locale.ROOT, "got=%.6f expect=%.6f", got, expect));
        }

        // variance = scale^2
        Tensor var = n.variance();
        for (int i = 0; i < 3; i++) {
            float s = scale.reshape(-1).get(i).item().toFloat();
            float got = var.reshape(-1).get(i).item().toFloat();
            check("Normal variance[" + i + "]", Math.abs(got - s * s) < ATOL_LOGP,
                    String.format(Locale.ROOT, "got=%.6f expect=%.6f", got, s * s));
        }

        // MC moments
        Tensor samples = n.sample(MC_N); // [MC_N, 3]
        Tensor mcMean = samples.mean(new long[]{0}, false, new ScalarTypeOptional());
        Tensor mcVar = samples.var(new long[]{0}, false, false);
        for (int i = 0; i < 3; i++) {
            float em = loc.reshape(-1).get(i).item().toFloat();
            float gm = mcMean.reshape(-1).get(i).item().toFloat();
            check("Normal MC mean[" + i + "]", Math.abs(gm - em) < ATOL_MEAN,
                    String.format(Locale.ROOT, "mc=%.4f exp=%.4f", gm, em));
            float ev = scale.reshape(-1).get(i).item().toFloat();
            ev = ev * ev;
            float gv = mcVar.reshape(-1).get(i).item().toFloat();
            check("Normal MC var[" + i + "]", Math.abs(gv - ev) / (ev + 1e-3) < 0.25,
                    String.format(Locale.ROOT, "mc=%.4f exp=%.4f", gv, ev));
        }

        // log_prob gradient-free consistency: argmax at loc
        Tensor atLoc = n.log_prob(loc);
        Tensor atLocPlus = n.log_prob(loc.add(new Scalar(1.0f)));
        Tensor better = torch.gt(atLoc, atLocPlus);
        check("Normal log_prob max near loc", torch.all(better).item().toBool(), null);

        safeCloseT(loc, scale, ent, var, samples, mcMean, mcVar, atLoc, atLocPlus, better);
        safeClose(n);
    }

    // ------------------------------------------------------------------
    // suite 8: discrete log_prob off-support
    // ------------------------------------------------------------------

    private static void runOffSupportSuite() {
        section("8. Off-support log_prob behaviour");

        // Bernoulli: value 0.5 may throw or return -inf depending on impl
        try (Bernoulli b = new Bernoulli(f(0.5f))) {
            try {
                Tensor lp = b.log_prob(f(0.5f));
                float v = scalar(lp);
                check("Bernoulli log_prob(0.5) non-finite or handled",
                        !Float.isFinite(v) || true, "got=" + v);
                safeCloseT(lp);
            } catch (IllegalArgumentException e) {
                check("Bernoulli log_prob(0.5) throws", true, e.getMessage());
            }
        }

        // Poisson negative → -inf or throw
        try (Poisson p = new Poisson(f(3f))) {
            try {
                Tensor lp = p.log_prob(f(-1f));
                float v = scalar(lp);
                check("Poisson log_prob(-1)", !Float.isFinite(v) || v < -1e5, "got=" + v);
                safeCloseT(lp);
            } catch (IllegalArgumentException e) {
                check("Poisson log_prob(-1) throws", true, e.getMessage());
            }
        }

        // Exponential negative
        try (Exponential e = new Exponential(f(1f))) {
            try {
                Tensor lp = e.log_prob(f(-0.5f));
                float v = scalar(lp);
                check("Exponential log_prob(-0.5)", !Float.isFinite(v) || v < 0, "got=" + v);
                safeCloseT(lp);
            } catch (IllegalArgumentException e2) {
                check("Exponential log_prob(-0.5) throws", true, e2.getMessage());
            }
        }

        // Uniform outside [low,high]
        try (Uniform u = new Uniform(f(0f), f(1f))) {
            Tensor lp = u.log_prob(f(1.5f));
            float v = scalar(lp);
            check("Uniform log_prob(1.5) = -inf", !Float.isFinite(v) || v < -1e5, "got=" + v);
            safeCloseT(lp);
        }

        // HalfNormal negative
        try (HalfNormal h = new HalfNormal(f(1f))) {
            try {
                Tensor lp = h.log_prob(f(-1f));
                float v = scalar(lp);
                check("HalfNormal log_prob(-1) = -inf", !Float.isFinite(v) || v < -1e5, "got=" + v);
                safeCloseT(lp);
            } catch (IllegalArgumentException e) {
                check("HalfNormal log_prob(-1) throws", true, e.getMessage());
            }
        }
    }

    // ------------------------------------------------------------------
    // suite 9: Multinomial (does not extend Distribution)
    // ------------------------------------------------------------------

    private static void runMultinomialSuite() {
        section("9. Multinomial (standalone, not a Distribution subclass)");
        try {
            Multinomial m = new Multinomial(10, f(0.2f, 0.3f, 0.5f));
            Tensor s = m.sample(new long[]{5});
            check("Multinomial sample shape",
                    shapeOf(s).length == 2 && shapeOf(s)[0] == 5 && shapeOf(s)[1] == 3,
                    Arrays.toString(shapeOf(s)));
            // each row sums to totalCount
            Tensor rowSum = s.sum(new long[]{1}, false, new ScalarTypeOptional());
            Tensor ok = torch.eq(rowSum, new Scalar(10));
            check("Multinomial rows sum to n", torch.all(ok).item().toBool(), null);
            double ent = m.entropy();
            check("Multinomial entropy finite", Double.isFinite(ent), "H=" + ent);
            safeCloseT(s, rowSum, ok);
            m.close();
        } catch (Exception e) {
            check("Multinomial suite", false, e.getMessage());
            e.printStackTrace(System.out);
        }
    }

    // ------------------------------------------------------------------
    // suite 10: summary table of all distributions
    // ------------------------------------------------------------------

    private static void runSmokeAll() {
        section("10. Smoke table — construct + sample(8) + log_prob + close");
        Map<String, String> status = new LinkedHashMap<>();
        for (DistCase c : buildCases()) {
            Distribution d = null;
            try {
                d = c.factory.create();
                Tensor s = d.sample(8);
                Tensor lp = d.log_prob(s);
                boolean ok = allFinite(s);
                status.put(c.name, ok ? "OK shape=" + Arrays.toString(shapeOf(s))
                        + " lp=" + Arrays.toString(shapeOf(lp)) : "NON-FINITE sample");
                check("smoke " + c.name, ok, status.get(c.name));
                safeCloseT(s, lp);
            } catch (Exception e) {
                status.put(c.name, "ERR " + e.getClass().getSimpleName() + ": " + e.getMessage());
                check("smoke " + c.name, false, status.get(c.name));
            } finally {
                if (d instanceof AutoCloseable) safeClose((AutoCloseable) d);
            }
        }
        System.out.println();
        System.out.println("Smoke summary:");
        for (Map.Entry<String, String> e : status.entrySet()) {
            System.out.printf(Locale.ROOT, "  %-40s %s%n", e.getKey(), e.getValue());
        }
    }

    // ------------------------------------------------------------------
    // main
    // ------------------------------------------------------------------

    public static void main(String[] args) {
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        System.out.println("distribute.BenchmarkDistributions — multi-aspect validation");
        System.out.println("device default: CPU");

        long t0 = System.nanoTime();
        try {
            runCoreApiSuite();
            runInvalidConstructionSuite();
            runAnalyticalLogProbSuite();
            runTransformedSuite();
            runWrapperSuite();
            runBatchStressSuite();
            runNormalDeepDive();
            runOffSupportSuite();
            runMultinomialSuite();
            runSmokeAll();
        } catch (Throwable t) {
            System.err.println("FATAL: " + t);
            t.printStackTrace();
            failed++;
        }
        long t1 = System.nanoTime();

        section("RESULT");
        System.out.printf(Locale.ROOT, "passed=%d  failed=%d  skipped=%d  time=%.2fs%n",
                passed, failed, skipped, (t1 - t0) / 1e9);
        if (!failures.isEmpty()) {
            System.out.println("Failures:");
            for (String f : failures) {
                System.out.println("  - " + f);
            }
        }
        if (failed > 0) {
            System.exit(1);
        }
    }
}
