package org.bytedeco.pytorch.rl.agent;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Welford running mean / variance for observation normalization.
 *
 * <p>Tracks per-feature statistics over a stream of samples shaped
 * {@code [D]} or batches {@code [N, D]}. {@link #normalize} returns
 * {@code (x - mean) / (std + eps)} with optional absolute clip.
 */
public final class RunningMeanStd {
    private final long dim;
    private final double eps;
    private final float clip;
    private long count;
    private final double[] mean;
    private final double[] m2; // sum of squares of differences from the current mean

    public RunningMeanStd(long dim) {
        this(dim, 1e-8, 10.0f);
    }

    public RunningMeanStd(long dim, double eps, float clip) {
        this.dim = Math.max(1, dim);
        this.eps = eps;
        this.clip = clip;
        this.count = 0;
        this.mean = new double[(int) this.dim];
        this.m2 = new double[(int) this.dim];
    }

    public long dim() { return dim; }
    public long count() { return count; }

    /** Update from one sample {@code [D]}. */
    public synchronized void update(float[] sample) {
        if (sample == null || sample.length == 0) return;
        count += 1;
        long n = count;
        int d = (int) Math.min(dim, sample.length);
        for (int i = 0; i < d; i++) {
            double delta = sample[i] - mean[i];
            mean[i] += delta / n;
            double delta2 = sample[i] - mean[i];
            m2[i] += delta * delta2;
        }
    }

    public synchronized void updateBatch(float[][] samples) {
        if (samples == null) return;
        for (float[] s : samples) update(s);
    }

    /**
     * Update from a tensor shaped {@code [D]}, {@code [1,D]}, or {@code [N,D]}.
     * This is the method {@link PPOAgent} calls.
     */
    public synchronized void update(Tensor x) {
        if (x == null || !x.defined()) return;
        Tensor t = x;
        if (t.dim() == 0) return;
        if (t.dim() == 1) {
            int d = (int) Math.min(dim, t.numel());
            float[] s = new float[d];
            for (int i = 0; i < d; i++) s[i] = t.select(0, i).item().toFloat();
            update(s);
            return;
        }
        // Leading batch dim
        long n = t.size(0);
        for (long row = 0; row < n; row++) {
            Tensor r = t.select(0, row).reshape(-1);
            int d = (int) Math.min(dim, r.numel());
            float[] s = new float[d];
            for (int i = 0; i < d; i++) s[i] = r.select(0, i).item().toFloat();
            update(s);
        }
    }

    /** {@code (x - mean) / std}, broadcast over leading dims. Identity if count &lt; 2. */
    public synchronized Tensor normalize(Tensor x) {
        if (count < 2) return x;
        float[] meanF = new float[(int) dim];
        float[] stdF = new float[(int) dim];
        for (int i = 0; i < dim; i++) {
            meanF[i] = (float) mean[i];
            double var = count > 1 ? m2[i] / (count - 1) : 0.0;
            stdF[i] = (float) Math.sqrt(Math.max(var, 0.0) + eps);
        }
        Tensor meanT = tensor(meanF);
        Tensor stdT = tensor(stdF);
        Tensor out = x.sub(meanT).div(stdT);
        if (clip > 0) {
            out = out.clamp(new ScalarOptional(new Scalar(-clip)),
                    new ScalarOptional(new Scalar(clip)));
        }
        return out;
    }

    public synchronized double[] meanCopy() {
        return mean.clone();
    }

    public synchronized double[] stdCopy() {
        double[] s = new double[(int) dim];
        for (int i = 0; i < dim; i++) {
            double var = count > 1 ? m2[i] / (count - 1) : 0.0;
            s[i] = Math.sqrt(Math.max(var, 0.0) + eps);
        }
        return s;
    }
}
