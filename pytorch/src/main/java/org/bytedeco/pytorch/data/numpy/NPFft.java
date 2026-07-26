package org.bytedeco.pytorch.data.numpy;

/**
 * NumPy-style FFT helpers (pure Java radix-2 Cooley–Tukey + Bluestein for general n).
 * Complex values are stored as interleaved [re, im] along a trailing size-2 axis
 * <em>or</em> as separate real/imag pair returns — here we use dual NDArray returns
 * {@code (real, imag)} for complex results, matching a simple binding-friendly API.
 *
 * <p>For real-input transforms that return complex, methods return {@code NDArray[2] = {re, im}}.
 * For real-output inverse rfft, a single real {@link NDArray} is returned.
 */
public final class NPFft {
    private NPFft() {}

    public static NDArray[] fft(NDArray a) { return fft(a, null, -1); }

    public static NDArray[] fft(NDArray a, Integer n, int axis) {
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        int len = n == null ? (int) a.shape[ax] : n;
        NDArray work = ensureAxisLen(a, ax, len);
        return transformAlong(work, ax, false);
    }

    public static NDArray[] ifft(NDArray a) { return ifft(a, null, -1); }

    /** Inverse: interpret a as real part only (imag zero) unless pair form used. */
    public static NDArray[] ifft(NDArray a, Integer n, int axis) {
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        int len = n == null ? (int) a.shape[ax] : n;
        NDArray work = ensureAxisLen(a, ax, len);
        NDArray[] out = transformAlong(work, ax, true);
        // scale 1/n
        double scale = 1.0 / len;
        out[0] = NPMath.multiply(out[0], scale);
        out[1] = NPMath.multiply(out[1], scale);
        return out;
    }

    /** Complex input fft: re/im same shape. */
    public static NDArray[] fft(NDArray re, NDArray im, Integer n, int axis) {
        int ax = NPArrayUtil.normalizeAxis(axis, re.shape.length);
        int len = n == null ? (int) re.shape[ax] : n;
        NDArray r = ensureAxisLen(re, ax, len);
        NDArray i = ensureAxisLen(im, ax, len);
        return transformAlong(r, i, ax, false);
    }

    public static NDArray[] ifft(NDArray re, NDArray im, Integer n, int axis) {
        int ax = NPArrayUtil.normalizeAxis(axis, re.shape.length);
        int len = n == null ? (int) re.shape[ax] : n;
        NDArray r = ensureAxisLen(re, ax, len);
        NDArray i = ensureAxisLen(im, ax, len);
        NDArray[] out = transformAlong(r, i, ax, true);
        double scale = 1.0 / len;
        out[0] = NPMath.multiply(out[0], scale);
        out[1] = NPMath.multiply(out[1], scale);
        return out;
    }

    public static NDArray[] rfft(NDArray a, Integer n, int axis) {
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        int len = n == null ? (int) a.shape[ax] : n;
        NDArray[] full = fft(a, len, ax);
        // keep 0..len/2 inclusive
        int keep = len / 2 + 1;
        return new NDArray[]{sliceAxis(full[0], ax, 0, keep), sliceAxis(full[1], ax, 0, keep)};
    }

    public static NDArray[] rfft(NDArray a) { return rfft(a, null, -1); }

    public static NDArray irfft(NDArray re, NDArray im, Integer n, int axis) {
        int ax = NPArrayUtil.normalizeAxis(axis, re.shape.length);
        int outLen = n == null ? 2 * ((int) re.shape[ax] - 1) : n;
        // build full Hermitian spectrum
        long[] fullShape = re.shape.clone();
        fullShape[ax] = outLen;
        NDArray fullRe = new NDArray(DType.FLOAT64, fullShape);
        NDArray fullIm = new NDArray(DType.FLOAT64, fullShape);
        int half = (int) re.shape[ax];
        // copy positive freqs
        copyAxisRange(re, fullRe, ax, 0, half, 0);
        copyAxisRange(im, fullIm, ax, 0, half, 0);
        // mirror
        for (int k = 1; k < half && outLen - k < outLen; k++) {
            // full[n-k] = conj(full[k])
            mirrorConj(fullRe, fullIm, ax, k, outLen - k);
        }
        NDArray[] inv = ifft(fullRe, fullIm, outLen, ax);
        return inv[0]; // real part
    }

    public static NDArray irfft(NDArray re, NDArray im) { return irfft(re, im, null, -1); }

    public static NDArray[] fft2(NDArray a) {
        NDArray[] r1 = fft(a, null, -1);
        NDArray[] r2re = fft(r1[0], r1[1], null, -2);
        return r2re;
    }

    public static NDArray[] ifft2(NDArray re, NDArray im) {
        NDArray[] r1 = ifft(re, im, null, -1);
        return ifft(r1[0], r1[1], null, -2);
    }

    public static NDArray[] fftn(NDArray a) {
        NDArray re = a;
        NDArray im = NP.zeros(DType.FLOAT64, a.shape);
        for (int ax = a.shape.length - 1; ax >= 0; ax--) {
            NDArray[] r = fft(re, im, null, ax);
            re = r[0]; im = r[1];
        }
        return new NDArray[]{re, im};
    }

    public static NDArray[] ifftn(NDArray re, NDArray im) {
        NDArray r = re, i = im;
        for (int ax = re.shape.length - 1; ax >= 0; ax--) {
            NDArray[] out = ifft(r, i, null, ax);
            r = out[0]; i = out[1];
        }
        return new NDArray[]{r, i};
    }

    public static NDArray fftshift(NDArray a, Integer axes) {
        if (axes == null) {
            NDArray out = a;
            for (int ax = 0; ax < a.shape.length; ax++) {
                int shift = (int) a.shape[ax] / 2;
                out = NPShape.roll(out, shift, ax);
            }
            return out;
        }
        int ax = NPArrayUtil.normalizeAxis(axes, a.shape.length);
        return NPShape.roll(a, (int) a.shape[ax] / 2, ax);
    }

    public static NDArray fftshift(NDArray a) { return fftshift(a, null); }

    public static NDArray ifftshift(NDArray a, Integer axes) {
        if (axes == null) {
            NDArray out = a;
            for (int ax = 0; ax < a.shape.length; ax++) {
                int shift = ((int) a.shape[ax] + 1) / 2;
                out = NPShape.roll(out, shift, ax);
            }
            return out;
        }
        int ax = NPArrayUtil.normalizeAxis(axes, a.shape.length);
        return NPShape.roll(a, ((int) a.shape[ax] + 1) / 2, ax);
    }

    public static NDArray ifftshift(NDArray a) { return ifftshift(a, null); }

    // ---- core transform -----------------------------------------------------

    private static NDArray[] transformAlong(NDArray re, int axis, boolean inverse) {
        NDArray im = new NDArray(DType.FLOAT64, re.shape);
        return transformAlong(re, im, axis, inverse);
    }

    private static NDArray[] transformAlong(NDArray re, NDArray im, int axis, boolean inverse) {
        int n = (int) re.shape[axis];
        NDArray outRe = new NDArray(DType.FLOAT64, re.shape);
        NDArray outIm = new NDArray(DType.FLOAT64, re.shape);
        long[] st = NPArrayUtil.stridesOf(re.shape);
        long otherN = re.size / n;
        long[] otherShape = new long[Math.max(0, re.shape.length - 1)];
        int k = 0;
        for (int d = 0; d < re.shape.length; d++) if (d != axis) otherShape[k++] = re.shape[d];
        long[] otherSt = otherShape.length == 0 ? new long[0] : NPArrayUtil.stridesOf(otherShape);
        int[] idx = new int[re.shape.length];
        double[] ar = new double[n];
        double[] ai = new double[n];
        for (int o = 0; o < otherN; o++) {
            int p = 0;
            for (int d = 0; d < re.shape.length; d++) {
                if (d == axis) idx[d] = 0;
                else {
                    idx[d] = otherShape.length == 0 ? 0 : (int) ((o / otherSt[p]) % otherShape[p]);
                    p++;
                }
            }
            for (int i = 0; i < n; i++) {
                idx[axis] = i;
                int flat = NPArrayUtil.ravel(idx, st);
                ar[i] = re.getDouble(flat);
                ai[i] = im.getDouble(flat);
            }
            fft1d(ar, ai, inverse);
            for (int i = 0; i < n; i++) {
                idx[axis] = i;
                int flat = NPArrayUtil.ravel(idx, st);
                outRe.setDouble(flat, ar[i]);
                outIm.setDouble(flat, ai[i]);
            }
        }
        return new NDArray[]{outRe, outIm};
    }

    /** In-place 1D FFT on re/im arrays of length n. */
    static void fft1d(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        if (n <= 1) return;
        if (Integer.bitCount(n) == 1) {
            radix2(re, im, inverse);
        } else {
            bluestein(re, im, inverse);
        }
    }

    private static void radix2(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        // bit reverse
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                double tr = re[i]; re[i] = re[j]; re[j] = tr;
                double ti = im[i]; im[i] = im[j]; im[j] = ti;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2 * Math.PI / len * (inverse ? 1 : -1);
            double wlenRe = Math.cos(ang), wlenIm = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double wr = 1, wi = 0;
                for (int j = 0; j < len / 2; j++) {
                    int u = i + j, v = i + j + len / 2;
                    double tr = wr * re[v] - wi * im[v];
                    double ti = wr * im[v] + wi * re[v];
                    re[v] = re[u] - tr; im[v] = im[u] - ti;
                    re[u] += tr; im[u] += ti;
                    double nwr = wr * wlenRe - wi * wlenIm;
                    wi = wr * wlenIm + wi * wlenRe;
                    wr = nwr;
                }
            }
        }
    }

    private static void bluestein(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        int m = 1;
        while (m < 2 * n - 1) m <<= 1;
        double[] ar = new double[m], ai = new double[m];
        double[] br = new double[m], bi = new double[m];
        for (int i = 0; i < n; i++) {
            double angle = Math.PI * i * i / n * (inverse ? 1 : -1);
            double cr = Math.cos(angle), ci = Math.sin(angle);
            ar[i] = re[i] * cr - im[i] * ci;
            ai[i] = re[i] * ci + im[i] * cr;
        }
        br[0] = 1; bi[0] = 0;
        for (int i = 1; i < n; i++) {
            double angle = Math.PI * i * i / n * (inverse ? -1 : 1);
            br[i] = br[m - i] = Math.cos(angle);
            bi[i] = bi[m - i] = Math.sin(angle);
        }
        radix2(ar, ai, false);
        radix2(br, bi, false);
        for (int i = 0; i < m; i++) {
            double tr = ar[i] * br[i] - ai[i] * bi[i];
            double ti = ar[i] * bi[i] + ai[i] * br[i];
            ar[i] = tr; ai[i] = ti;
        }
        radix2(ar, ai, true);
        for (int i = 0; i < m; i++) { ar[i] /= m; ai[i] /= m; }
        for (int i = 0; i < n; i++) {
            double angle = Math.PI * i * i / n * (inverse ? 1 : -1);
            double cr = Math.cos(angle), ci = Math.sin(angle);
            re[i] = ar[i] * cr - ai[i] * ci;
            im[i] = ar[i] * ci + ai[i] * cr;
        }
    }

    private static NDArray ensureAxisLen(NDArray a, int axis, int len) {
        if (a.shape[axis] == len) return a;
        long[] shape = a.shape.clone();
        shape[axis] = len;
        NDArray out = new NDArray(a.dtype, shape);
        long copy = Math.min(a.shape[axis], len);
        long[] aSt = NPArrayUtil.stridesOf(a.shape);
        long[] oSt = NPArrayUtil.stridesOf(shape);
        int[] idx = new int[a.shape.length];
        // zero already; copy overlapping
        long other = a.size / a.shape[axis];
        // iterate all positions with axis < copy
        for (int flat = 0; flat < a.size; flat++) {
            NPArrayUtil.fillMultiIndex(flat, a.shape, aSt, idx);
            if (idx[axis] >= copy) continue;
            out.setDouble(NPArrayUtil.ravel(idx, oSt), a.getDouble(flat));
        }
        return out;
    }

    private static NDArray sliceAxis(NDArray a, int axis, int start, int end) {
        long[] shape = a.shape.clone();
        shape[axis] = end - start;
        NDArray out = new NDArray(a.dtype, shape);
        long[] aSt = NPArrayUtil.stridesOf(a.shape);
        long[] oSt = NPArrayUtil.stridesOf(shape);
        int[] idx = new int[a.shape.length];
        for (int flat = 0; flat < out.size; flat++) {
            NPArrayUtil.fillMultiIndex(flat, shape, oSt, idx);
            int[] src = idx.clone();
            src[axis] = idx[axis] + start;
            out.setDouble(flat, a.getDouble(NPArrayUtil.ravel(src, aSt)));
        }
        return out;
    }

    private static void copyAxisRange(NDArray src, NDArray dst, int axis, int srcStart, int len, int dstStart) {
        long[] sSt = NPArrayUtil.stridesOf(src.shape);
        long[] dSt = NPArrayUtil.stridesOf(dst.shape);
        // build reduced iteration over non-axis + range
        long[] shape = src.shape.clone();
        shape[axis] = len;
        int[] idx = new int[src.shape.length];
        long[] st = NPArrayUtil.stridesOf(shape);
        long n = NPArrayUtil.numel(shape);
        for (int flat = 0; flat < n; flat++) {
            NPArrayUtil.fillMultiIndex(flat, shape, st, idx);
            int[] sIdx = idx.clone(); sIdx[axis] = idx[axis] + srcStart;
            int[] dIdx = idx.clone(); dIdx[axis] = idx[axis] + dstStart;
            dst.setDouble(NPArrayUtil.ravel(dIdx, dSt), src.getDouble(NPArrayUtil.ravel(sIdx, sSt)));
        }
    }

    private static void mirrorConj(NDArray re, NDArray im, int axis, int from, int to) {
        long[] st = NPArrayUtil.stridesOf(re.shape);
        long other = re.size / re.shape[axis];
        long[] otherShape = new long[Math.max(0, re.shape.length - 1)];
        int k = 0;
        for (int d = 0; d < re.shape.length; d++) if (d != axis) otherShape[k++] = re.shape[d];
        long[] otherSt = otherShape.length == 0 ? new long[0] : NPArrayUtil.stridesOf(otherShape);
        int[] idx = new int[re.shape.length];
        for (int o = 0; o < other; o++) {
            int p = 0;
            for (int d = 0; d < re.shape.length; d++) {
                if (d == axis) continue;
                idx[d] = otherShape.length == 0 ? 0 : (int) ((o / otherSt[p]) % otherShape[p]);
                p++;
            }
            idx[axis] = from;
            int f = NPArrayUtil.ravel(idx, st);
            idx[axis] = to;
            int t = NPArrayUtil.ravel(idx, st);
            re.setDouble(t, re.getDouble(f));
            im.setDouble(t, -im.getDouble(f));
        }
    }
}
