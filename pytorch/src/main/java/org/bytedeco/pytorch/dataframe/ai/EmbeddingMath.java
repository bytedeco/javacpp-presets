package org.bytedeco.pytorch.dataframe.ai;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

/**
 * Shared vector math for embedding backends (L2, cosine, mean-pool, hashing).
 */
public final class EmbeddingMath {
    private EmbeddingMath() {}

    public static float[] l2Normalize(float[] v) {
        if (v == null) return null;
        double sum = 0;
        for (float x : v) sum += (double) x * x;
        double n = Math.sqrt(sum);
        if (n < 1e-12) return Arrays.copyOf(v, v.length);
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = (float) (v[i] / n);
        return out;
    }

    public static double cosine(float[] a, float[] b) {
        if (a == null || b == null) return Double.NaN;
        int n = Math.min(a.length, b.length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        double d = Math.sqrt(na) * Math.sqrt(nb);
        return d < 1e-12 ? 0.0 : dot / d;
    }

    public static double dot(float[] a, float[] b) {
        if (a == null || b == null) return Double.NaN;
        int n = Math.min(a.length, b.length);
        double s = 0;
        for (int i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    public static float[] meanPool(float[][] frames) {
        if (frames == null || frames.length == 0) return new float[0];
        int dim = 0;
        for (float[] f : frames) if (f != null) { dim = f.length; break; }
        if (dim == 0) return new float[0];
        float[] out = new float[dim];
        int count = 0;
        for (float[] f : frames) {
            if (f == null) continue;
            count++;
            for (int i = 0; i < dim && i < f.length; i++) out[i] += f[i];
        }
        if (count == 0) return out;
        for (int i = 0; i < dim; i++) out[i] /= count;
        return out;
    }

    public static float[] meanPool(List<float[]> frames) {
        if (frames == null || frames.isEmpty()) return new float[0];
        return meanPool(frames.toArray(new float[0][]));
    }

    /** Feature-hash a UTF-8 string into a fixed-dim dense vector (Murmur-ish). */
    public static float[] hashEmbedText(String text, int dim) {
        float[] out = new float[dim];
        if (text == null || text.isEmpty() || dim <= 0) return out;
        byte[] bytes = text.toLowerCase().getBytes(StandardCharsets.UTF_8);
        // unigrams
        for (int i = 0; i < bytes.length; i++) {
            int h = mix(bytes[i] & 0xFF, i);
            out[floorMod(h, dim)] += 1.0f;
            out[floorMod(h * 0x9E3779B9, dim)] += 0.5f;
        }
        // bigrams
        for (int i = 0; i + 1 < bytes.length; i++) {
            int h = mix(((bytes[i] & 0xFF) << 8) | (bytes[i + 1] & 0xFF), i * 31);
            out[floorMod(h, dim)] += 0.75f;
        }
        // char n-gram skip
        for (int i = 0; i + 2 < bytes.length; i += 2) {
            int h = mix(bytes[i] * 131 + bytes[i + 2], 17);
            out[floorMod(h, dim)] += 0.35f;
        }
        return l2Normalize(out);
    }

    /** Hash-embed a float signal (audio samples / flattened image) via stats + bins. */
    public static float[] hashEmbedSignal(float[] signal, int dim) {
        float[] out = new float[Math.max(1, dim)];
        if (signal == null || signal.length == 0) return out;
        // global stats in first slots
        double sum = 0, sum2 = 0, min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (float v : signal) {
            sum += v; sum2 += v * v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        int n = signal.length;
        double mean = sum / n;
        double var = Math.max(0, sum2 / n - mean * mean);
        out[0 % dim] += (float) mean;
        out[1 % dim] += (float) Math.sqrt(var);
        out[2 % dim] += (float) min;
        out[3 % dim] += (float) max;
        out[4 % dim] += (float) Math.log1p(n);

        // histogram into remaining dims
        int bins = Math.max(8, dim - 8);
        for (float v : signal) {
            double t = (v - min) / (max - min + 1e-12);
            int b = Math.min(bins - 1, Math.max(0, (int) (t * bins)));
            out[(8 + b) % dim] += 1.0f;
        }
        // strided samples
        int step = Math.max(1, n / Math.max(1, dim));
        for (int i = 0, k = 0; i < n && k < dim; i += step, k++) {
            out[k] += signal[i] * 0.1f;
        }
        return l2Normalize(out);
    }

    /** Project RGB float image (HxWx3 interleaved) to embedding. */
    public static float[] hashEmbedImageRgb(float[] rgb, int height, int width, int dim) {
        if (rgb == null) return new float[dim];
        // multi-scale block means + color hist
        float[] out = new float[dim];
        int blocks = 4;
        int bw = Math.max(1, width / blocks);
        int bh = Math.max(1, height / blocks);
        int slot = 0;
        for (int by = 0; by < blocks; by++) {
            for (int bx = 0; bx < blocks; bx++) {
                double r = 0, g = 0, b = 0; int c = 0;
                for (int y = by * bh; y < Math.min(height, (by + 1) * bh); y++) {
                    for (int x = bx * bw; x < Math.min(width, (bx + 1) * bw); x++) {
                        int i = (y * width + x) * 3;
                        if (i + 2 >= rgb.length) continue;
                        r += rgb[i]; g += rgb[i + 1]; b += rgb[i + 2]; c++;
                    }
                }
                if (c > 0) {
                    out[slot++ % dim] += (float) (r / c);
                    out[slot++ % dim] += (float) (g / c);
                    out[slot++ % dim] += (float) (b / c);
                }
            }
        }
        // coarse color histogram
        for (int i = 0; i + 2 < rgb.length; i += 3) {
            int ri = Math.min(7, (int) (rgb[i] * 8));
            int gi = Math.min(7, (int) (rgb[i + 1] * 8));
            int bi = Math.min(7, (int) (rgb[i + 2] * 8));
            out[floorMod(ri * 64 + gi * 8 + bi, dim)] += 1.0f;
        }
        out[0] += (float) Math.log1p(height * width);
        return l2Normalize(out);
    }

    public static float[] zeros(int dim) { return new float[Math.max(0, dim)]; }

    public static float[] concat(float[] a, float[] b) {
        if (a == null) return b == null ? null : Arrays.copyOf(b, b.length);
        if (b == null) return Arrays.copyOf(a, a.length);
        float[] out = new float[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /** Deterministic mix — stable across JVMs for same inputs. */
    public static int mix(int z, int salt) {
        int h = z ^ salt;
        h ^= (h >>> 16);
        h *= 0x7feb352d;
        h ^= (h >>> 15);
        h *= 0x846ca68b;
        h ^= (h >>> 16);
        return h;
    }

    public static int floorMod(int x, int m) {
        if (m <= 0) return 0;
        int r = x % m;
        return r < 0 ? r + m : r;
    }

    public static float[] ensureDim(float[] v, int dim) {
        if (v == null) return zeros(dim);
        if (v.length == dim) return v;
        float[] out = new float[dim];
        System.arraycopy(v, 0, out, 0, Math.min(v.length, dim));
        return out;
    }
}
