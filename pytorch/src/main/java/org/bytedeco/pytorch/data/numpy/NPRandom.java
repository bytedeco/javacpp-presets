package org.bytedeco.pytorch.data.numpy;

import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;

/**
 * NumPy-style random sampling. Uses {@link ThreadLocalRandom} by default;
 * call {@link #seed(long)} to install a deterministic {@link java.util.Random}.
 */
public final class NPRandom {
    private static volatile RandomGenerator RNG = ThreadLocalRandom.current();

    private NPRandom() {}

    public static void seed(long seed) {
        RNG = new java.util.Random(seed);
    }

    public static void seed() {
        RNG = ThreadLocalRandom.current();
    }

    private static RandomGenerator rng() {
        RandomGenerator r = RNG;
        // ThreadLocalRandom.current() must be re-read per call on that path
        if (r instanceof ThreadLocalRandom) return ThreadLocalRandom.current();
        return r;
    }

    public static NDArray rand(long... shape) {
        NDArray a = new NDArray(DType.FLOAT64, shape);
        RandomGenerator r = rng();
        for (int i = 0; i < a.size; i++) a.setDouble(i, r.nextDouble());
        return a;
    }

    public static NDArray randn(long... shape) {
        NDArray a = new NDArray(DType.FLOAT64, shape);
        RandomGenerator r = rng();
        for (int i = 0; i < a.size; i++) a.setDouble(i, r.nextGaussian());
        return a;
    }

    public static NDArray random(long... shape) { return rand(shape); }

    public static NDArray ranf(long... shape) { return rand(shape); }

    public static NDArray sample(long... shape) { return rand(shape); }

    public static NDArray uniform(double low, double high, long... size) {
        NDArray a = new NDArray(DType.FLOAT64, size);
        RandomGenerator r = rng();
        double w = high - low;
        for (int i = 0; i < a.size; i++) a.setDouble(i, low + w * r.nextDouble());
        return a;
    }

    public static NDArray normal(double loc, double scale, long... size) {
        NDArray a = new NDArray(DType.FLOAT64, size);
        RandomGenerator r = rng();
        for (int i = 0; i < a.size; i++) a.setDouble(i, loc + scale * r.nextGaussian());
        return a;
    }

    public static NDArray standard_normal(long... size) { return randn(size); }

    public static NDArray randint(int low, int high, long... size) {
        if (high <= low) throw new IllegalArgumentException("high must be > low");
        NDArray a = new NDArray(DType.INT64, size);
        RandomGenerator r = rng();
        int span = high - low;
        for (int i = 0; i < a.size; i++) a.setLong(i, low + r.nextInt(span));
        return a;
    }

    public static NDArray randint(int high, long... size) { return randint(0, high, size); }

    public static NDArray random_integers(int low, int high, long... size) {
        return randint(low, high + 1, size);
    }

    public static NDArray randn_like(NDArray a) { return randn(a.shape); }

    public static NDArray rand_like(NDArray a) { return rand(a.shape); }

    public static NDArray permutation(NDArray a) {
        NDArray out = NPShape.ravel(NPShape.copy(a));
        shuffle(out);
        return out;
    }

    public static NDArray permutation(int n) {
        NDArray a = NP.arange(n);
        // arange returns float64 by default — use int indices
        long[] idx = new long[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        NDArray out = new NDArray(idx, DType.INT64);
        shuffle(out);
        return out;
    }

    /** In-place shuffle of flat array contents. */
    public static void shuffle(NDArray x) {
        RandomGenerator r = rng();
        int n = (int) x.size;
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            double ti = x.getDouble(i);
            x.setDouble(i, x.getDouble(j));
            x.setDouble(j, ti);
        }
    }

    public static NDArray choice(NDArray a, Integer size, boolean replace, NDArray p) {
        NDArray flat = NPShape.ravel(a);
        int n = (int) flat.size;
        int outN = size == null ? 1 : size;
        if (!replace && outN > n) throw new IllegalArgumentException("cannot take more than population");
        NDArray out = new NDArray(flat.dtype, outN);
        RandomGenerator r = rng();
        if (p == null) {
            if (replace) {
                for (int i = 0; i < outN; i++) out.setDouble(i, flat.getDouble(r.nextInt(n)));
            } else {
                NDArray idx = permutation(n);
                for (int i = 0; i < outN; i++) out.setDouble(i, flat.getDouble((int) idx.getLong(i)));
            }
            return size == null ? NPShape.reshape(out) : out; // scalar-ish length-1
        }
        // weighted
        double[] cdf = new double[n];
        double s = 0;
        for (int i = 0; i < n; i++) { s += p.getDouble(i); cdf[i] = s; }
        if (Math.abs(s - 1.0) > 1e-6) {
            for (int i = 0; i < n; i++) cdf[i] /= s;
        }
        boolean[] used = replace ? null : new boolean[n];
        for (int k = 0; k < outN; k++) {
            int pick = -1;
            for (int attempt = 0; attempt < 10000; attempt++) {
                double u = r.nextDouble();
                int lo = 0, hi = n - 1;
                while (lo < hi) {
                    int mid = (lo + hi) >>> 1;
                    if (cdf[mid] < u) lo = mid + 1; else hi = mid;
                }
                pick = lo;
                if (replace || !used[pick]) break;
            }
            if (!replace) used[pick] = true;
            out.setDouble(k, flat.getDouble(pick));
        }
        return out;
    }

    public static NDArray choice(NDArray a, int size) { return choice(a, size, true, null); }

    public static NDArray choice(NDArray a) { return choice(a, null, true, null); }

    public static NDArray choice(int n, int size, boolean replace) {
        return choice(permutation(n), size, replace, null); // wasteful but ok
    }

    public static double random() { return rng().nextDouble(); }

    public static NDArray beta(double a, double b, long... size) {
        // via gamma ratio
        NDArray out = new NDArray(DType.FLOAT64, size);
        for (int i = 0; i < out.size; i++) {
            double x = gammaSample(a);
            double y = gammaSample(b);
            out.setDouble(i, x / (x + y));
        }
        return out;
    }

    public static NDArray exponential(double scale, long... size) {
        NDArray out = new NDArray(DType.FLOAT64, size);
        RandomGenerator r = rng();
        for (int i = 0; i < out.size; i++) out.setDouble(i, -scale * Math.log(1 - r.nextDouble()));
        return out;
    }

    public static NDArray poisson(double lam, long... size) {
        NDArray out = new NDArray(DType.INT64, size);
        for (int i = 0; i < out.size; i++) out.setLong(i, poissonSample(lam));
        return out;
    }

    public static NDArray binomial(int n, double p, long... size) {
        NDArray out = new NDArray(DType.INT64, size);
        RandomGenerator r = rng();
        for (int i = 0; i < out.size; i++) {
            int c = 0;
            for (int k = 0; k < n; k++) if (r.nextDouble() < p) c++;
            out.setLong(i, c);
        }
        return out;
    }

    public static NDArray geometric(double p, long... size) {
        NDArray out = new NDArray(DType.INT64, size);
        RandomGenerator r = rng();
        for (int i = 0; i < out.size; i++) {
            out.setLong(i, (long) Math.ceil(Math.log(1 - r.nextDouble()) / Math.log(1 - p)));
        }
        return out;
    }

    public static NDArray logistic(double loc, double scale, long... size) {
        NDArray out = new NDArray(DType.FLOAT64, size);
        RandomGenerator r = rng();
        for (int i = 0; i < out.size; i++) {
            double u = r.nextDouble();
            out.setDouble(i, loc + scale * Math.log(u / (1 - u)));
        }
        return out;
    }

    public static NDArray laplace(double loc, double scale, long... size) {
        NDArray out = new NDArray(DType.FLOAT64, size);
        RandomGenerator r = rng();
        for (int i = 0; i < out.size; i++) {
            double u = r.nextDouble() - 0.5;
            out.setDouble(i, loc - scale * Math.signum(u) * Math.log(1 - 2 * Math.abs(u)));
        }
        return out;
    }

    public static NDArray rayleigh(double scale, long... size) {
        NDArray out = new NDArray(DType.FLOAT64, size);
        RandomGenerator r = rng();
        for (int i = 0; i < out.size; i++) {
            out.setDouble(i, scale * Math.sqrt(-2 * Math.log(1 - r.nextDouble())));
        }
        return out;
    }

    public static NDArray gumbel(double loc, double scale, long... size) {
        NDArray out = new NDArray(DType.FLOAT64, size);
        RandomGenerator r = rng();
        for (int i = 0; i < out.size; i++) {
            out.setDouble(i, loc - scale * Math.log(-Math.log(1 - r.nextDouble())));
        }
        return out;
    }

    public static NDArray lognormal(double mean, double sigma, long... size) {
        NDArray out = new NDArray(DType.FLOAT64, size);
        RandomGenerator r = rng();
        for (int i = 0; i < out.size; i++) {
            out.setDouble(i, Math.exp(mean + sigma * r.nextGaussian()));
        }
        return out;
    }

    public static NDArray chisquare(double df, long... size) {
        NDArray out = new NDArray(DType.FLOAT64, size);
        for (int i = 0; i < out.size; i++) out.setDouble(i, 2 * gammaSample(df / 2.0));
        return out;
    }

    public static NDArray gamma(double shape, double scale, long... size) {
        NDArray out = new NDArray(DType.FLOAT64, size);
        for (int i = 0; i < out.size; i++) out.setDouble(i, scale * gammaSample(shape));
        return out;
    }

    public static NDArray multinomial(int n, double[] pvals, long... size) {
        // returns shape size + (pvals.length,)
        long[] shape = new long[size.length + 1];
        System.arraycopy(size, 0, shape, 0, size.length);
        shape[size.length] = pvals.length;
        NDArray out = new NDArray(DType.INT64, shape);
        long rows = NPArrayUtil.numel(size.length == 0 ? new long[]{1} : size);
        if (size.length == 0) rows = 1;
        int k = pvals.length;
        for (int r = 0; r < rows; r++) {
            int remain = n;
            double pLeft = 1;
            for (int j = 0; j < k; j++) {
                double pj = pLeft > 0 ? pvals[j] / pLeft : 0;
                int c = j == k - 1 ? remain : binomialOne(remain, pj);
                out.setLong((int) (r * k + j), c);
                remain -= c;
                pLeft -= pvals[j];
            }
        }
        return out;
    }

    private static int binomialOne(int n, double p) {
        RandomGenerator r = rng();
        int c = 0;
        for (int i = 0; i < n; i++) if (r.nextDouble() < p) c++;
        return c;
    }

    /** Marsaglia-Tsang for shape >= 1; boost for shape < 1. */
    private static double gammaSample(double shape) {
        RandomGenerator r = rng();
        if (shape <= 0) throw new IllegalArgumentException("shape must be > 0");
        if (shape < 1) {
            return gammaSample(shape + 1) * Math.pow(r.nextDouble(), 1.0 / shape);
        }
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        while (true) {
            double x, v;
            do {
                x = r.nextGaussian();
                v = 1 + c * x;
            } while (v <= 0);
            v = v * v * v;
            double u = r.nextDouble();
            if (u < 1 - 0.0331 * x * x * x * x) return d * v;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
        }
    }

    private static long poissonSample(double lam) {
        RandomGenerator r = rng();
        if (lam < 30) {
            double L = Math.exp(-lam);
            long k = 0;
            double p = 1;
            do {
                k++;
                p *= r.nextDouble();
            } while (p > L);
            return k - 1;
        }
        // normal approx for large lam
        return Math.max(0, Math.round(lam + Math.sqrt(lam) * r.nextGaussian()));
    }
}
