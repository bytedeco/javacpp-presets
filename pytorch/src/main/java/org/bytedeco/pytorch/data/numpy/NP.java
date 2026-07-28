package org.bytedeco.pytorch.data.numpy;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.plot.*;

import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * NumPy-like API over {@link NDArray}: factories, ufuncs, reductions, shape ops,
 * linalg/FFT/random entry points, {@code .npy}/{@code .npz} I/O and LibTorch
 * {@link Tensor} conversion.
 *
 * <pre>
 *   NDArray a = NP.zeros(2, 3);
 *   NDArray b = NP.sin(NP.add(a, 1.0));
 *   double s = NP.sum(b);
 *   Tensor t = NP.toTensor(b);
 * </pre>
 *
 * <p>Heavy implementations live in {@link NPMath}, {@link NPReduce}, {@link NPShape},
 * {@link NPLinalg}, {@link NPFft}, {@link NPRandom}; this class re-exports the public surface.
 */
public final class NP {
    private NP() {}

    /** Nested linalg namespace: {@code NP.Linalg.inv(a)}. */
    public static final class Linalg {
        private Linalg() {}
        public static NDArray inv(NDArray a) { return NPLinalg.inv(a); }
        public static NDArray pinv(NDArray a) { return NPLinalg.pinv(a); }
        public static NDArray pinv(NDArray a, double rcond) { return NPLinalg.pinv(a, rcond); }
        public static NDArray matrix_power(NDArray a, int n) { return NPLinalg.matrix_power(a, n); }
        public static double det(NDArray a) { return NPLinalg.det(a); }
        public static NDArray solve(NDArray a, NDArray b) { return NPLinalg.solve(a, b); }
        public static NDArray[] lstsq(NDArray a, NDArray b) { return NPLinalg.lstsq(a, b); }
        public static NDArray cholesky(NDArray a) { return NPLinalg.cholesky(a); }
        public static NDArray[] qr(NDArray a) { return NPLinalg.qr(a); }
        public static NDArray[] svd(NDArray a) { return NPLinalg.svd(a); }
        public static NDArray[] svd(NDArray a, boolean fullMatrices, boolean computeUv) {
            return NPLinalg.svd(a, fullMatrices, computeUv);
        }
        public static NDArray[] eig(NDArray a) { return NPLinalg.eig(a); }
        public static NDArray[] eigh(NDArray a) { return NPLinalg.eigh(a); }
        public static NDArray norm(NDArray x) { return NPLinalg.norm(x); }
        public static NDArray norm(NDArray x, Integer axis) { return NPLinalg.norm(x, axis); }
        public static NDArray norm(NDArray x, String ord, Integer axis, boolean keepdims) {
            return NPLinalg.norm(x, ord, axis, keepdims);
        }
        public static NDArray matrix_rank(NDArray a) {
            NDArray[] usv = NPLinalg.svd(a, false, false);
            NDArray S = usv[1];
            double tol = 1e-9;
            int rank = 0;
            for (int i = 0; i < S.size; i++) if (S.getDouble(i) > tol) rank++;
            NDArray out = new NDArray(DType.INT64);
            out.setLong(0, rank);
            return out;
        }
        public static NDArray multi_dot(NDArray... arrays) {
            if (arrays == null || arrays.length == 0) throw new IllegalArgumentException("need arrays");
            NDArray r = arrays[0];
            for (int i = 1; i < arrays.length; i++) r = NPLinalg.matmul(r, arrays[i]);
            return r;
        }
        public static NDArray tensorinv(NDArray a, int ind) {
            // reshape to square then inv then reshape back — simplified for ind leading dims
            long prod = 1;
            for (int i = 0; i < ind; i++) prod *= a.shape[i];
            long back = a.size / prod;
            if (prod != back) throw new IllegalArgumentException("tensorinv requires square unfolding");
            NDArray flat = NPShape.reshape(a, prod, prod);
            return NPShape.reshape(NPLinalg.inv(flat), a.shape);
        }
        public static NDArray tensorsolve(NDArray a, NDArray b) {
            long prod = b.size;
            long back = a.size / prod;
            NDArray flat = NPShape.reshape(a, prod, back);
            NDArray x = NPLinalg.solve(NPShape.transpose(flat), // Ax=b with A shaped...
                    NPShape.reshape(b, (int) b.size));
            // simplified: treat a as (b.size, x.size)
            flat = NPShape.reshape(a, b.size, a.size / b.size);
            x = NPLinalg.solve(flat, NPShape.ravel(b));
            return x;
        }
    }

    /** Nested FFT namespace: {@code NP.Fft.fft(a)}. */
    public static final class Fft {
        private Fft() {}
        public static NDArray[] fft(NDArray a) { return NPFft.fft(a); }
        public static NDArray[] fft(NDArray a, Integer n, int axis) { return NPFft.fft(a, n, axis); }
        public static NDArray[] fft(NDArray re, NDArray im, Integer n, int axis) { return NPFft.fft(re, im, n, axis); }
        public static NDArray[] ifft(NDArray a) { return NPFft.ifft(a); }
        public static NDArray[] ifft(NDArray a, Integer n, int axis) { return NPFft.ifft(a, n, axis); }
        public static NDArray[] ifft(NDArray re, NDArray im, Integer n, int axis) { return NPFft.ifft(re, im, n, axis); }
        public static NDArray[] rfft(NDArray a) { return NPFft.rfft(a); }
        public static NDArray[] rfft(NDArray a, Integer n, int axis) { return NPFft.rfft(a, n, axis); }
        public static NDArray irfft(NDArray re, NDArray im) { return NPFft.irfft(re, im); }
        public static NDArray irfft(NDArray re, NDArray im, Integer n, int axis) { return NPFft.irfft(re, im, n, axis); }
        public static NDArray[] fft2(NDArray a) { return NPFft.fft2(a); }
        public static NDArray[] ifft2(NDArray re, NDArray im) { return NPFft.ifft2(re, im); }
        public static NDArray[] fftn(NDArray a) { return NPFft.fftn(a); }
        public static NDArray[] ifftn(NDArray re, NDArray im) { return NPFft.ifftn(re, im); }
        public static NDArray fftshift(NDArray a) { return NPFft.fftshift(a); }
        public static NDArray ifftshift(NDArray a) { return NPFft.ifftshift(a); }
    }

    /** Nested random namespace: {@code NP.Random.randn(2,3)}. */
    public static final class Random {
        private Random() {}
        public static void seed(long s) { NPRandom.seed(s); }
        public static void seed() { NPRandom.seed(); }
        public static NDArray rand(long... shape) { return NPRandom.rand(shape); }
        public static NDArray randn(long... shape) { return NPRandom.randn(shape); }
        public static NDArray random(long... shape) { return NPRandom.random(shape); }
        public static NDArray uniform(double low, double high, long... size) { return NPRandom.uniform(low, high, size); }
        public static NDArray normal(double loc, double scale, long... size) { return NPRandom.normal(loc, scale, size); }
        public static NDArray standard_normal(long... size) { return NPRandom.standard_normal(size); }
        public static NDArray randint(int low, int high, long... size) { return NPRandom.randint(low, high, size); }
        public static NDArray randint(int high, long... size) { return NPRandom.randint(high, size); }
        public static NDArray permutation(NDArray a) { return NPRandom.permutation(a); }
        public static NDArray permutation(int n) { return NPRandom.permutation(n); }
        public static void shuffle(NDArray x) { NPRandom.shuffle(x); }
        public static NDArray choice(NDArray a, int size) { return NPRandom.choice(a, size); }
        public static NDArray choice(NDArray a, Integer size, boolean replace, NDArray p) {
            return NPRandom.choice(a, size, replace, p);
        }
        public static NDArray beta(double a, double b, long... size) { return NPRandom.beta(a, b, size); }
        public static NDArray exponential(double scale, long... size) { return NPRandom.exponential(scale, size); }
        public static NDArray poisson(double lam, long... size) { return NPRandom.poisson(lam, size); }
        public static NDArray binomial(int n, double p, long... size) { return NPRandom.binomial(n, p, size); }
        public static NDArray geometric(double p, long... size) { return NPRandom.geometric(p, size); }
        public static NDArray logistic(double loc, double scale, long... size) { return NPRandom.logistic(loc, scale, size); }
        public static NDArray laplace(double loc, double scale, long... size) { return NPRandom.laplace(loc, scale, size); }
        public static NDArray rayleigh(double scale, long... size) { return NPRandom.rayleigh(scale, size); }
        public static NDArray gumbel(double loc, double scale, long... size) { return NPRandom.gumbel(loc, scale, size); }
        public static NDArray lognormal(double mean, double sigma, long... size) { return NPRandom.lognormal(mean, sigma, size); }
        public static NDArray chisquare(double df, long... size) { return NPRandom.chisquare(df, size); }
        public static NDArray gamma(double shape, double scale, long... size) { return NPRandom.gamma(shape, scale, size); }
        public static NDArray multinomial(int n, double[] pvals, long... size) { return NPRandom.multinomial(n, pvals, size); }
        public static double random() { return NPRandom.random(); }
    }

    /** Nested polynomial namespace: {@code NP.Poly.polyfit(x,y,2)}. */
    public static final class Poly {
        private Poly() {}
        public static NDArray polyfit(NDArray x, NDArray y, int deg) { return NPPoly.polyfit(x, y, deg); }
        public static NDArray polyval(NDArray p, NDArray x) { return NPPoly.polyval(p, x); }
        public static double polyval(NDArray p, double x) { return NPPoly.polyval(p, x); }
        public static NDArray roots(NDArray p) { return NPPoly.roots(p); }
        public static NDArray polyadd(NDArray a, NDArray b) { return NPPoly.polyadd(a, b); }
        public static NDArray polysub(NDArray a, NDArray b) { return NPPoly.polysub(a, b); }
        public static NDArray polymul(NDArray a, NDArray b) { return NPPoly.polymul(a, b); }
        public static NDArray polyder(NDArray p) { return NPPoly.polyder(p); }
        public static NDArray polyder(NDArray p, int m) { return NPPoly.polyder(p, m); }
        public static NDArray polyint(NDArray p) { return NPPoly.polyint(p); }
        public static NDArray polyint(NDArray p, int m, double[] k) { return NPPoly.polyint(p, m, k); }
        public static NPPoly.Poly1d poly1d(NDArray c) { return NPPoly.poly1d(c); }
        public static NPPoly.Poly1d poly1d(double[] c) { return NPPoly.poly1d(c); }
    }

    /** Nested plotting namespace: {@code NP.Plot.plot(x,y).show()}. */
    public static final class Plot {
        private Plot() {}
        public static LineChart plot(NDArray y) { return NPPlot.plot(y); }
        public static LineChart plot(NDArray x, NDArray y) { return NPPlot.plot(x, y); }
        public static LineChart plot(NDArray x, NDArray y, String label) { return NPPlot.plot(x, y, label); }
        public static LineChart plot(NDArray x, NDArray Y, String[] labels) { return NPPlot.plot(x, Y, labels); }
        public static ScatterChart scatter(NDArray x, NDArray y) { return NPPlot.scatter(x, y); }
        public static HistogramChart hist(NDArray data, int bins) { return NPPlot.hist(data, bins); }
        public static HistogramChart hist(NDArray data) { return NPPlot.hist(data); }
        public static BarChart bar(NDArray values) { return NPPlot.bar(values); }
        public static BarChart bar(String[] cats, NDArray values) { return NPPlot.bar(cats, values); }
        public static PieChart pie(String[] labels, NDArray values) { return NPPlot.pie(labels, values); }
        public static BoxChart boxplot(NDArray values) { return NPPlot.boxplot(values); }
        public static HeatmapChart imshow(NDArray a) { return NPPlot.imshow(a); }
        public static HeatmapChart heatmap(NDArray a) { return NPPlot.heatmap(a); }
        public static HeatmapChart corrplot(NDArray a) { return NPPlot.corrplot(a); }
        public static void show() { NPPlot.show(); }
        public static void savefig(String path) throws Exception { NPPlot.savefig(path); }
        public static BaseChart title(String t) { return NPPlot.title(t); }
        public static BaseChart xlabel(String s) { return NPPlot.xlabel(s); }
        public static BaseChart ylabel(String s) { return NPPlot.ylabel(s); }
        public static BaseChart legend(boolean on) { return NPPlot.legend(on); }
        public static BaseChart grid(boolean on) { return NPPlot.grid(on); }
        public static BaseChart last() { return NPPlot.last(); }
    }

    /** Nested PCA namespace: {@code NP.PCA.fitTransform(X, 2)}. */
    public static final class PCA {
        private PCA() {}
        public static NPPCA.Result fitTransform(NDArray X) { return NPPCA.fitTransform(X); }
        public static NPPCA.Result fitTransform(NDArray X, Integer nComponents) { return NPPCA.fitTransform(X, nComponents); }
        public static NDArray transform(NDArray X, NPPCA.Result model) { return NPPCA.transform(X, model); }
        public static NDArray inverseTransform(NDArray Xpca, NPPCA.Result model) { return NPPCA.inverseTransform(Xpca, model); }
    }

    /** Nested masked-array namespace: {@code NP.Ma.masked_where(cond, a)}. */
    public static final class Ma {
        private Ma() {}
        public static MaskedArray masked_array(NDArray data, NDArray mask) { return MaskedArray.masked_array(data, mask); }
        public static MaskedArray masked_array(NDArray data, NDArray mask, double fill) { return MaskedArray.masked_array(data, mask, fill); }
        public static MaskedArray masked_where(NDArray cond, NDArray data) { return MaskedArray.masked_where(cond, data); }
        public static MaskedArray masked_equal(NDArray data, double v) { return MaskedArray.masked_equal(data, v); }
        public static MaskedArray masked_invalid(NDArray data) { return MaskedArray.masked_invalid(data); }
        public static MaskedArray masked_greater(NDArray data, double v) { return MaskedArray.masked_greater(data, v); }
        public static MaskedArray masked_less(NDArray data, double v) { return MaskedArray.masked_less(data, v); }
    }

    // ---- factories ----------------------------------------------------------

    public static NDArray zeros(DType dtype, long... shape) {
        return new NDArray(dtype, shape);
    }

    public static NDArray zeros(long... shape) {
        return zeros(DType.FLOAT64, shape);
    }

    public static NDArray ones(DType dtype, long... shape) {
        NDArray a = new NDArray(dtype, shape);
        for (int i = 0; i < a.size; i++) a.setDouble(i, 1.0);
        return a;
    }

    public static NDArray ones(long... shape) {
        return ones(DType.FLOAT64, shape);
    }

    public static NDArray empty(DType dtype, long... shape) {
        return new NDArray(dtype, shape);
    }

    public static NDArray empty(long... shape) {
        return empty(DType.FLOAT64, shape);
    }

    public static NDArray full(DType dtype, double value, long... shape) {
        NDArray a = new NDArray(dtype, shape);
        for (int i = 0; i < a.size; i++) a.setDouble(i, value);
        return a;
    }

    public static NDArray full(double value, long... shape) {
        return full(DType.FLOAT64, value, shape);
    }

    public static NDArray zeros_like(NDArray a) { return zeros(a.dtype, a.shape); }
    public static NDArray ones_like(NDArray a) { return ones(a.dtype, a.shape); }
    public static NDArray empty_like(NDArray a) { return empty(a.dtype, a.shape); }
    public static NDArray full_like(NDArray a, double value) { return full(a.dtype, value, a.shape); }

    public static NDArray eye(int n) { return eye(n, n, 0); }
    public static NDArray eye(int n, int m) { return eye(n, m, 0); }

    public static NDArray eye(int n, int m, int k) {
        NDArray a = zeros(DType.FLOAT64, n, m);
        for (int i = 0; i < n; i++) {
            int j = i + k;
            if (j >= 0 && j < m) a.setDouble(i * m + j, 1.0);
        }
        return a;
    }

    public static NDArray identity(int n) { return eye(n); }

    public static NDArray arange(double start, double stop, double step, DType dtype) {
        int n = (int) Math.max(0, Math.ceil((stop - start) / step));
        NDArray a = new NDArray(dtype, n);
        for (int i = 0; i < n; i++) a.setDouble(i, start + i * step);
        return a;
    }

    public static NDArray arange(double stop) { return arange(0, stop, 1.0, DType.FLOAT64); }
    public static NDArray arange(double start, double stop) { return arange(start, stop, 1.0, DType.FLOAT64); }
    public static NDArray arange(double start, double stop, double step) { return arange(start, stop, step, DType.FLOAT64); }

    public static NDArray linspace(double start, double stop, int num) {
        if (num <= 0) return new NDArray(DType.FLOAT64, 0);
        if (num == 1) return array(new double[]{start});
        NDArray a = new NDArray(DType.FLOAT64, num);
        double step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) a.setDouble(i, start + i * step);
        return a;
    }

    public static NDArray logspace(double start, double stop, int num) {
        return logspace(start, stop, num, 10.0);
    }

    public static NDArray logspace(double start, double stop, int num, double base) {
        NDArray exp = linspace(start, stop, num);
        return NPMath.power(full(base, exp.shape), exp);
    }

    public static NDArray geomspace(double start, double stop, int num) {
        if (start == 0 || stop == 0) throw new IllegalArgumentException("geomspace bounds must be non-zero");
        double logStart = Math.log(Math.abs(start));
        double logStop = Math.log(Math.abs(stop));
        NDArray a = linspace(logStart, logStop, num);
        NDArray out = NPMath.exp(a);
        if (start < 0) out = NPMath.negative(out);
        return out;
    }

    public static NDArray array(double[] data, long... shape) {
        return new NDArray(data, shape.length > 0 ? shape : new long[]{data.length});
    }

    public static NDArray array(float[] data, long... shape) {
        return new NDArray(data, shape.length > 0 ? shape : new long[]{data.length});
    }

    public static NDArray array(long[] data, long... shape) {
        return new NDArray(data, DType.INT64, shape.length > 0 ? shape : new long[]{data.length});
    }

    public static NDArray array(long[] data, DType dtype, long... shape) {
        return new NDArray(data, dtype, shape.length > 0 ? shape : new long[]{data.length});
    }

    public static NDArray array(int[] data, long... shape) {
        long[] longs = new long[data.length];
        for (int i = 0; i < data.length; i++) longs[i] = data[i];
        return new NDArray(longs, DType.INT32, shape.length > 0 ? shape : new long[]{data.length});
    }

    public static NDArray asarray(Object x) { return NPArrayUtil.asArray(x); }

    public static NDArray rand(long... shape) { return NPRandom.rand(shape); }
    public static NDArray randn(long... shape) { return NPRandom.randn(shape); }

    // ---- copy / cast --------------------------------------------------------

    public static NDArray copy(NDArray a) { return NPShape.copy(a); }
    public static NDArray astype(NDArray a, DType dtype) { return NPShape.astype(a, dtype); }

    // ---- unary math (re-export) ---------------------------------------------

    public static NDArray abs(NDArray a) { return NPMath.abs(a); }
    public static NDArray fabs(NDArray a) { return NPMath.fabs(a); }
    public static NDArray sqrt(NDArray a) { return NPMath.sqrt(a); }
    public static NDArray square(NDArray a) { return NPMath.square(a); }
    public static NDArray cbrt(NDArray a) { return NPMath.cbrt(a); }
    public static NDArray exp(NDArray a) { return NPMath.exp(a); }
    public static NDArray exp2(NDArray a) { return NPMath.exp2(a); }
    public static NDArray expm1(NDArray a) { return NPMath.expm1(a); }
    public static NDArray log(NDArray a) { return NPMath.log(a); }
    public static NDArray log2(NDArray a) { return NPMath.log2(a); }
    public static NDArray log10(NDArray a) { return NPMath.log10(a); }
    public static NDArray log1p(NDArray a) { return NPMath.log1p(a); }
    public static NDArray sign(NDArray a) { return NPMath.sign(a); }
    public static NDArray ceil(NDArray a) { return NPMath.ceil(a); }
    public static NDArray floor(NDArray a) { return NPMath.floor(a); }
    public static NDArray trunc(NDArray a) { return NPMath.trunc(a); }
    public static NDArray rint(NDArray a) { return NPMath.rint(a); }
    public static NDArray round(NDArray a) { return NPMath.round(a); }
    public static NDArray negative(NDArray a) { return NPMath.negative(a); }
    public static NDArray neg(NDArray a) { return NPMath.neg(a); }
    public static NDArray positive(NDArray a) { return NPMath.positive(a); }
    public static NDArray reciprocal(NDArray a) { return NPMath.reciprocal(a); }
    public static NDArray isfinite(NDArray a) { return NPMath.isfinite(a); }
    public static NDArray isinf(NDArray a) { return NPMath.isinf(a); }
    public static NDArray isnan(NDArray a) { return NPMath.isnan(a); }
    public static NDArray isreal(NDArray a) { return NPMath.isreal(a); }
    public static NDArray imag(NDArray a) { return NPMath.imag(a); }
    public static NDArray real(NDArray a) { return NPMath.real(a); }
    public static NDArray conj(NDArray a) { return NPMath.conj(a); }
    public static NDArray signbit(NDArray a) { return NPMath.signbit(a); }
    public static NDArray relu(NDArray a) { return NPMath.relu(a); }
    public static NDArray leaky_relu(NDArray a, double alpha) { return NPMath.leaky_relu(a, alpha); }
    public static NDArray sigmoid(NDArray a) { return NPMath.sigmoid(a); }

    public static NDArray sin(NDArray a) { return NPMath.sin(a); }
    public static NDArray cos(NDArray a) { return NPMath.cos(a); }
    public static NDArray tan(NDArray a) { return NPMath.tan(a); }
    public static NDArray arcsin(NDArray a) { return NPMath.arcsin(a); }
    public static NDArray asin(NDArray a) { return NPMath.asin(a); }
    public static NDArray arccos(NDArray a) { return NPMath.arccos(a); }
    public static NDArray acos(NDArray a) { return NPMath.acos(a); }
    public static NDArray arctan(NDArray a) { return NPMath.arctan(a); }
    public static NDArray atan(NDArray a) { return NPMath.atan(a); }
    public static NDArray sinh(NDArray a) { return NPMath.sinh(a); }
    public static NDArray cosh(NDArray a) { return NPMath.cosh(a); }
    public static NDArray tanh(NDArray a) { return NPMath.tanh(a); }
    public static NDArray arcsinh(NDArray a) { return NPMath.arcsinh(a); }
    public static NDArray asinh(NDArray a) { return NPMath.asinh(a); }
    public static NDArray arccosh(NDArray a) { return NPMath.arccosh(a); }
    public static NDArray acosh(NDArray a) { return NPMath.acosh(a); }
    public static NDArray arctanh(NDArray a) { return NPMath.arctanh(a); }
    public static NDArray atanh(NDArray a) { return NPMath.atanh(a); }
    public static NDArray radians(NDArray a) { return NPMath.radians(a); }
    public static NDArray degrees(NDArray a) { return NPMath.degrees(a); }
    public static NDArray deg2rad(NDArray a) { return NPMath.deg2rad(a); }
    public static NDArray rad2deg(NDArray a) { return NPMath.rad2deg(a); }
    public static NDArray arctan2(NDArray y, NDArray x) { return NPMath.arctan2(y, x); }
    public static NDArray atan2(NDArray y, NDArray x) { return NPMath.atan2(y, x); }

    // ---- binary -------------------------------------------------------------

    public static NDArray add(NDArray x1, NDArray x2) { return NPMath.add(x1, x2); }
    public static NDArray add(NDArray a, double s) { return NPMath.add(a, s); }
    public static NDArray subtract(NDArray x1, NDArray x2) { return NPMath.subtract(x1, x2); }
    public static NDArray sub(NDArray x1, NDArray x2) { return NPMath.sub(x1, x2); }
    public static NDArray subtract(NDArray a, double s) { return NPMath.subtract(a, s); }
    public static NDArray sub(NDArray a, double s) { return NPMath.sub(a, s); }
    public static NDArray multiply(NDArray x1, NDArray x2) { return NPMath.multiply(x1, x2); }
    public static NDArray mul(NDArray x1, NDArray x2) { return NPMath.mul(x1, x2); }
    public static NDArray multiply(NDArray a, double s) { return NPMath.multiply(a, s); }
    public static NDArray mul(NDArray a, double s) { return NPMath.mul(a, s); }
    public static NDArray divide(NDArray x1, NDArray x2) { return NPMath.divide(x1, x2); }
    public static NDArray div(NDArray x1, NDArray x2) { return NPMath.div(x1, x2); }
    public static NDArray divide(NDArray a, double s) { return NPMath.divide(a, s); }
    public static NDArray div(NDArray a, double s) { return NPMath.div(a, s); }
    public static NDArray true_divide(NDArray x1, NDArray x2) { return NPMath.true_divide(x1, x2); }
    public static NDArray floor_divide(NDArray x1, NDArray x2) { return NPMath.floor_divide(x1, x2); }
    public static NDArray power(NDArray x1, NDArray x2) { return NPMath.power(x1, x2); }
    public static NDArray power(NDArray a, double exp) { return NPMath.power(a, exp); }
    public static NDArray pow(NDArray x1, NDArray x2) { return NPMath.pow(x1, x2); }
    public static NDArray mod(NDArray x1, NDArray x2) { return NPMath.mod(x1, x2); }
    public static NDArray remainder(NDArray x1, NDArray x2) { return NPMath.remainder(x1, x2); }
    public static NDArray fmod(NDArray x1, NDArray x2) { return NPMath.fmod(x1, x2); }
    public static NDArray maximum(NDArray x1, NDArray x2) { return NPMath.maximum(x1, x2); }
    public static NDArray maximum(NDArray a, double s) { return NPMath.maximum(a, s); }
    public static NDArray minimum(NDArray x1, NDArray x2) { return NPMath.minimum(x1, x2); }
    public static NDArray minimum(NDArray a, double s) { return NPMath.minimum(a, s); }
    public static NDArray fmax(NDArray x1, NDArray x2) { return NPMath.fmax(x1, x2); }
    public static NDArray fmin(NDArray x1, NDArray x2) { return NPMath.fmin(x1, x2); }
    public static NDArray hypot(NDArray x1, NDArray x2) { return NPMath.hypot(x1, x2); }
    public static NDArray copysign(NDArray x1, NDArray x2) { return NPMath.copysign(x1, x2); }
    public static NDArray gcd(NDArray x1, NDArray x2) { return NPMath.gcd(x1, x2); }
    public static NDArray lcm(NDArray x1, NDArray x2) { return NPMath.lcm(x1, x2); }

    public static NDArray equal(NDArray x1, NDArray x2) { return NPMath.equal(x1, x2); }
    public static NDArray not_equal(NDArray x1, NDArray x2) { return NPMath.not_equal(x1, x2); }
    public static NDArray greater(NDArray x1, NDArray x2) { return NPMath.greater(x1, x2); }
    public static NDArray greater_equal(NDArray x1, NDArray x2) { return NPMath.greater_equal(x1, x2); }
    public static NDArray less(NDArray x1, NDArray x2) { return NPMath.less(x1, x2); }
    public static NDArray less_equal(NDArray x1, NDArray x2) { return NPMath.less_equal(x1, x2); }
    public static NDArray logical_and(NDArray x1, NDArray x2) { return NPMath.logical_and(x1, x2); }
    public static NDArray logical_or(NDArray x1, NDArray x2) { return NPMath.logical_or(x1, x2); }
    public static NDArray logical_xor(NDArray x1, NDArray x2) { return NPMath.logical_xor(x1, x2); }
    public static NDArray logical_not(NDArray x) { return NPMath.logical_not(x); }

    public static NDArray clip(NDArray a, double min, double max) { return NPMath.clip(a, min, max); }
    public static NDArray clip(NDArray a, NDArray min, NDArray max) { return NPMath.clip(a, min, max); }
    public static NDArray where(NDArray cond, NDArray x, NDArray y) { return NPMath.where(cond, x, y); }
    public static NDArray[] where(NDArray cond) { return NPReduce.where(cond); }
    public static NDArray heaviside(NDArray x1, NDArray x2) { return NPMath.heaviside(x1, x2); }
    public static NDArray nan_to_num(NDArray a) { return NPMath.nan_to_num(a); }

    // ---- reductions ---------------------------------------------------------

    public static double sum(NDArray a) { return NPReduce.sum(a); }
    public static NDArray sum(NDArray a, Integer axis, boolean keepdims) { return NPReduce.sum(a, axis, keepdims); }
    public static NDArray sum(NDArray a, int axis) { return NPReduce.sum(a, axis, false); }

    public static double mean(NDArray a) { return NPReduce.mean(a); }
    public static NDArray mean(NDArray a, Integer axis, boolean keepdims) { return NPReduce.mean(a, axis, keepdims); }
    public static NDArray mean(NDArray a, int axis) { return NPReduce.mean(a, axis, false); }

    public static double prod(NDArray a) { return NPReduce.prod(a); }
    public static NDArray prod(NDArray a, Integer axis, boolean keepdims) { return NPReduce.prod(a, axis, keepdims); }

    public static double max(NDArray a) { return NPReduce.max(a); }
    public static NDArray max(NDArray a, Integer axis, boolean keepdims) { return NPReduce.max(a, axis, keepdims); }
    public static NDArray max(NDArray a, int axis) { return NPReduce.max(a, axis, false); }
    public static double amax(NDArray a) { return max(a); }
    public static NDArray amax(NDArray a, Integer axis, boolean keepdims) { return max(a, axis, keepdims); }

    public static double min(NDArray a) { return NPReduce.min(a); }
    public static NDArray min(NDArray a, Integer axis, boolean keepdims) { return NPReduce.min(a, axis, keepdims); }
    public static NDArray min(NDArray a, int axis) { return NPReduce.min(a, axis, false); }
    public static double amin(NDArray a) { return min(a); }

    public static double var(NDArray a) { return NPReduce.var(a); }
    public static double var(NDArray a, int ddof) { return NPReduce.var(a, ddof); }
    public static NDArray var(NDArray a, Integer axis, boolean keepdims) { return NPReduce.var(a, axis, keepdims); }
    public static NDArray var(NDArray a, Integer axis, boolean keepdims, int ddof) { return NPReduce.var(a, axis, keepdims, ddof); }

    public static double std(NDArray a) { return NPReduce.std(a); }
    public static double std(NDArray a, int ddof) { return NPReduce.std(a, ddof); }
    public static NDArray std(NDArray a, Integer axis, boolean keepdims) { return NPReduce.std(a, axis, keepdims); }
    public static NDArray std(NDArray a, Integer axis, boolean keepdims, int ddof) { return NPReduce.std(a, axis, keepdims, ddof); }

    public static double nansum(NDArray a) { return NPReduce.nansum(a); }
    public static double nanmean(NDArray a) { return NPReduce.nanmean(a); }

    public static int argmax(NDArray a) { return NPReduce.argmax(a); }
    public static NDArray argmax(NDArray a, Integer axis, boolean keepdims) { return NPReduce.argmax(a, axis, keepdims); }
    public static NDArray argmax(NDArray a, int axis) { return NPReduce.argmax(a, axis, false); }

    public static int argmin(NDArray a) { return NPReduce.argmin(a); }
    public static NDArray argmin(NDArray a, Integer axis, boolean keepdims) { return NPReduce.argmin(a, axis, keepdims); }
    public static NDArray argmin(NDArray a, int axis) { return NPReduce.argmin(a, axis, false); }

    public static boolean any(NDArray a) { return NPReduce.any(a); }
    public static NDArray any(NDArray a, Integer axis, boolean keepdims) { return NPReduce.any(a, axis, keepdims); }
    public static boolean all(NDArray a) { return NPReduce.all(a); }
    public static NDArray all(NDArray a, Integer axis, boolean keepdims) { return NPReduce.all(a, axis, keepdims); }

    public static double median(NDArray a) { return NPReduce.median(a); }
    public static NDArray median(NDArray a, Integer axis, boolean keepdims) { return NPReduce.median(a, axis, keepdims); }
    public static double percentile(NDArray a, double q) { return NPReduce.percentile(a, q); }
    public static NDArray percentile(NDArray a, double q, Integer axis, boolean keepdims) {
        return NPReduce.percentile(a, q, axis, keepdims);
    }

    public static NDArray cumsum(NDArray a) { return NPReduce.cumsum(a, null); }
    public static NDArray cumsum(NDArray a, Integer axis) { return NPReduce.cumsum(a, axis); }
    public static NDArray cumprod(NDArray a) { return NPReduce.cumprod(a, null); }
    public static NDArray cumprod(NDArray a, Integer axis) { return NPReduce.cumprod(a, axis); }

    public static NDArray sort(NDArray a) { return NPReduce.sort(a, -1); }
    public static NDArray sort(NDArray a, Integer axis) { return NPReduce.sort(a, axis); }
    public static NDArray argsort(NDArray a) { return NPReduce.argsort(a, -1); }
    public static NDArray argsort(NDArray a, Integer axis) { return NPReduce.argsort(a, axis); }
    public static NDArray partition(NDArray a, int kth) { return NPReduce.partition(a, kth, -1); }
    public static NDArray partition(NDArray a, int kth, Integer axis) { return NPReduce.partition(a, kth, axis); }
    public static NDArray argpartition(NDArray a, int kth) { return NPReduce.argpartition(a, kth, -1); }
    public static NDArray argpartition(NDArray a, int kth, Integer axis) { return NPReduce.argpartition(a, kth, axis); }
    public static NDArray unique(NDArray a) { return NPReduce.unique(a); }
    public static NDArray searchsorted(NDArray a, NDArray v) { return NPReduce.searchsorted(a, v); }
    public static NDArray bincount(NDArray x) { return NPReduce.bincount(x); }
    public static NDArray bincount(NDArray x, NDArray weights, int minlength) { return NPReduce.bincount(x, weights, minlength); }
    public static NDArray[] histogram(NDArray a, int bins) { return NPReduce.histogram(a, bins); }
    public static NDArray digitize(NDArray x, NDArray bins) { return NPReduce.digitize(x, bins); }
    public static NDArray extract(NDArray condition, NDArray arr) { return NPReduce.extract(condition, arr); }
    public static NDArray[] nonzero(NDArray a) { return NPReduce.nonzero(a); }
    public static NDArray diff(NDArray a) { return NPReduce.diff(a); }
    public static NDArray diff(NDArray a, int n, Integer axis) { return NPReduce.diff(a, n, axis); }
    public static NDArray ediff1d(NDArray ary) { return NPReduce.ediff1d(ary); }
    public static NDArray setdiff1d(NDArray x, NDArray y) { return NPReduce.setdiff1d(x, y); }
    public static NDArray interp(NDArray x, NDArray xp, NDArray fp) { return NPReduce.interp(x, xp, fp); }
    public static NDArray cov(NDArray m) { return NPReduce.cov(m); }
    public static NDArray cov(NDArray m, boolean rowvar) { return NPReduce.cov(m, rowvar); }
    public static NDArray corrcoef(NDArray x) { return NPReduce.corrcoef(x); }
    public static NDArray convolve(NDArray a, NDArray v) { return NPReduce.convolve(a, v, "full"); }
    public static NDArray convolve(NDArray a, NDArray v, String mode) { return NPReduce.convolve(a, v, mode); }
    public static NDArray correlate(NDArray a, NDArray v) { return NPReduce.correlate(a, v, "valid"); }
    public static NDArray correlate(NDArray a, NDArray v, String mode) { return NPReduce.correlate(a, v, mode); }

    // ---- shape --------------------------------------------------------------

    public static NDArray reshape(NDArray a, long... newShape) { return NPShape.reshape(a, newShape); }
    public static NDArray ravel(NDArray a) { return NPShape.ravel(a); }
    public static NDArray flatten(NDArray a) { return NPShape.flatten(a); }
    public static NDArray transpose(NDArray a) { return NPShape.transpose(a); }
    public static NDArray transpose(NDArray a, int... axes) { return NPShape.transpose(a, axes); }
    public static NDArray swapaxes(NDArray a, int axis1, int axis2) { return NPShape.swapaxes(a, axis1, axis2); }
    public static NDArray moveaxis(NDArray a, int source, int destination) { return NPShape.moveaxis(a, source, destination); }
    public static NDArray expand_dims(NDArray a, int axis) { return NPShape.expand_dims(a, axis); }
    public static NDArray squeeze(NDArray a) { return NPShape.squeeze(a); }
    public static NDArray squeeze(NDArray a, Integer axis) { return NPShape.squeeze(a, axis); }
    public static NDArray flip(NDArray a) { return NPShape.flip(a, (Integer) null); }
    public static NDArray flip(NDArray a, int axis) { return NPShape.flip(a, axis); }
    public static NDArray fliplr(NDArray m) { return NPShape.fliplr(m); }
    public static NDArray flipud(NDArray m) { return NPShape.flipud(m); }
    public static NDArray rot90(NDArray m) { return NPShape.rot90(m); }
    public static NDArray rot90(NDArray m, int k) { return NPShape.rot90(m, k); }
    public static NDArray rot90(NDArray m, int k, int[] axes) { return NPShape.rot90(m, k, axes); }
    public static NDArray broadcast_to(NDArray a, long... shape) { return NPShape.broadcast_to(a, shape); }
    public static NDArray[] broadcast_arrays(NDArray... arrays) { return NPShape.broadcast_arrays(arrays); }
    public static NDArray atleast_1d(NDArray a) { return NPShape.atleast_1d(a); }
    public static NDArray atleast_2d(NDArray a) { return NPShape.atleast_2d(a); }
    public static NDArray atleast_3d(NDArray a) { return NPShape.atleast_3d(a); }
    public static NDArray concatenate(NDArray[] arrays, int axis) { return NPShape.concatenate(arrays, axis); }
    public static NDArray concatenate(NDArray a, NDArray b) { return NPShape.concatenate(a, b); }
    public static NDArray concatenate(NDArray a, NDArray b, int axis) { return NPShape.concatenate(a, b, axis); }
    public static NDArray stack(NDArray[] arrays, int axis) { return NPShape.stack(arrays, axis); }
    public static NDArray stack(NDArray a, NDArray b) { return NPShape.stack(a, b); }
    public static NDArray hstack(NDArray... arrays) { return NPShape.hstack(arrays); }
    public static NDArray vstack(NDArray... arrays) { return NPShape.vstack(arrays); }
    public static NDArray dstack(NDArray... arrays) { return NPShape.dstack(arrays); }
    public static NDArray[] split(NDArray ary, int sections, int axis) { return NPShape.split(ary, sections, axis); }
    public static NDArray[] array_split(NDArray ary, int sections, int axis) { return NPShape.array_split(ary, sections, axis); }
    public static NDArray[] array_split(NDArray ary, long[] indices, int axis) { return NPShape.array_split(ary, indices, axis); }
    public static NDArray[] hsplit(NDArray ary, int sections) { return NPShape.hsplit(ary, sections); }
    public static NDArray[] vsplit(NDArray ary, int sections) { return NPShape.vsplit(ary, sections); }
    public static NDArray repeat(NDArray a, int repeats) { return NPShape.repeat(a, repeats); }
    public static NDArray repeat(NDArray a, int repeats, Integer axis) { return NPShape.repeat(a, repeats, axis); }
    public static NDArray tile(NDArray A, long... reps) { return NPShape.tile(A, reps); }
    public static NDArray roll(NDArray a, int shift) { return NPShape.roll(a, shift); }
    public static NDArray roll(NDArray a, int shift, Integer axis) { return NPShape.roll(a, shift, axis); }
    public static NDArray[] meshgrid(NDArray... xi) { return NPShape.meshgrid(xi); }
    public static NDArray[] meshgrid(boolean indexingXy, NDArray... xi) { return NPShape.meshgrid(indexingXy, xi); }
    public static NDArray diag(NDArray v) { return NPShape.diag(v); }
    public static NDArray diag(NDArray v, int k) { return NPShape.diag(v, k); }
    public static NDArray diagonal(NDArray a) { return NPShape.diagonal(a, 0, 0, 1); }
    public static NDArray diagonal(NDArray a, int offset, int axis1, int axis2) {
        return NPShape.diagonal(a, offset, axis1, axis2);
    }
    public static NDArray ascontiguousarray(NDArray a) { return NPShape.ascontiguousarray(a); }
    public static NDArray asfortranarray(NDArray a) { return NPShape.asfortranarray(a); }
    public static NDArray as_strided(NDArray x, long[] shape, long[] strides) { return NPShape.as_strided(x, shape, strides); }
    public static NDArray as_strided(NDArray x, long[] shape, long[] strides, long offset) {
        return NPShape.as_strided(x, shape, strides, offset);
    }
    public static NDArray sliding_window_view(NDArray x, long... windowShape) {
        return NPShape.sliding_window_view(x, windowShape);
    }
    public static NDArray[] ogrid(NDArray... xi) { return NPShape.ogrid(xi); }
    public static NDArray[] mgrid(NDArray... xi) { return NPShape.mgrid(xi); }

    // ---- complex ------------------------------------------------------------

    public static NDArray complex(NDArray real, NDArray imag) { return NPComplex.complex(real, imag); }
    public static NDArray complex(NDArray real, NDArray imag, DType dtype) { return NPComplex.complex(real, imag, dtype); }
    public static NDArray complex(double re, double im) { return NPComplex.complex(re, im); }
    public static NDArray angle(NDArray a) { return NPComplex.angle(a); }
    public static NDArray absolute(NDArray a) { return NPComplex.absolute(a); }
    public static NDArray iscomplexobj(NDArray a) { return NPComplex.iscomplex(a); }

    // ---- bits ---------------------------------------------------------------

    public static NDArray packbits(NDArray a) { return NPBits.packbits(a); }
    public static NDArray packbits(NDArray a, Integer axis, String bitorder) { return NPBits.packbits(a, axis, bitorder); }
    public static NDArray unpackbits(NDArray a) { return NPBits.unpackbits(a); }
    public static NDArray unpackbits(NDArray a, Integer axis, Integer count, String bitorder) {
        return NPBits.unpackbits(a, axis, count, bitorder);
    }

    // ---- polynomial (also under NP.Poly) ------------------------------------

    public static NDArray polyfit(NDArray x, NDArray y, int deg) { return NPPoly.polyfit(x, y, deg); }
    public static NDArray polyval(NDArray p, NDArray x) { return NPPoly.polyval(p, x); }
    public static double polyval(NDArray p, double x) { return NPPoly.polyval(p, x); }
    public static NDArray roots(NDArray p) { return NPPoly.roots(p); }
    public static NDArray polyadd(NDArray a, NDArray b) { return NPPoly.polyadd(a, b); }
    public static NDArray polysub(NDArray a, NDArray b) { return NPPoly.polysub(a, b); }
    public static NDArray polymul(NDArray a, NDArray b) { return NPPoly.polymul(a, b); }
    public static NDArray polyder(NDArray p) { return NPPoly.polyder(p); }
    public static NDArray polyint(NDArray p) { return NPPoly.polyint(p); }
    public static NPPoly.Poly1d poly1d(NDArray c) { return NPPoly.poly1d(c); }
    public static NPPoly.Poly1d poly1d(double[] c) { return NPPoly.poly1d(c); }

    // ---- linalg top-level ---------------------------------------------------

    public static NDArray dot(NDArray a, NDArray b) { return NPLinalg.dot(a, b); }
    public static NDArray matmul(NDArray a, NDArray b) { return NPLinalg.matmul(a, b); }
    public static NDArray tensordot(NDArray a, NDArray b, int axes) { return NPLinalg.tensordot(a, b, axes); }
    public static double trace(NDArray a) { return NPLinalg.trace(a); }
    public static NDArray vander(NDArray x) { return NPLinalg.vander(x); }
    public static NDArray vander(NDArray x, Integer N, boolean increasing) { return NPLinalg.vander(x, N, increasing); }

    // ---- activations --------------------------------------------------------

    public static NDArray softmax(NDArray a) {
        double m = max(a);
        NDArray e = exp(sub(a, m));
        double s = sum(e);
        return div(e, s);
    }

    public static NDArray softmax(NDArray a, int axis) {
        if (a.shape.length != 2) return softmax(a);
        long rows = a.shape[0], cols = a.shape[1];
        NDArray out = new NDArray(a.dtype, a.shape);
        if (axis == 1 || axis == -1) {
            for (int i = 0; i < rows; i++) {
                double m = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < cols; j++) m = Math.max(m, a.getDouble((int) (i * cols + j)));
                double s = 0;
                for (int j = 0; j < cols; j++) {
                    double v = Math.exp(a.getDouble((int) (i * cols + j)) - m);
                    out.setDouble((int) (i * cols + j), v);
                    s += v;
                }
                for (int j = 0; j < cols; j++) {
                    int idx = (int) (i * cols + j);
                    out.setDouble(idx, out.getDouble(idx) / s);
                }
            }
        } else {
            for (int j = 0; j < cols; j++) {
                double m = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < rows; i++) m = Math.max(m, a.getDouble((int) (i * cols + j)));
                double s = 0;
                for (int i = 0; i < rows; i++) {
                    double v = Math.exp(a.getDouble((int) (i * cols + j)) - m);
                    out.setDouble((int) (i * cols + j), v);
                    s += v;
                }
                for (int i = 0; i < rows; i++) {
                    int idx = (int) (i * cols + j);
                    out.setDouble(idx, out.getDouble(idx) / s);
                }
            }
        }
        return out;
    }

    // ---- .npy I/O -----------------------------------------------------------

    public static void save(NDArray a, String path) throws IOException {
        NpyHeader headerObj = new NpyHeader(a.dtype, false, a.shape);
        String headerContent = headerObj.toHeaderString();

        int prefixLen = 10; // magic(6)+ver(2)+hlen(2)
        int targetLen = ((prefixLen + headerContent.length() + 1 + 63) / 64) * 64;
        int paddingLen = targetLen - prefixLen - headerContent.length() - 1;

        StringBuilder sb = new StringBuilder(headerContent);
        for (int i = 0; i < paddingLen; i++) sb.append(' ');
        sb.append('\n');
        byte[] headerBytes = sb.toString().getBytes(StandardCharsets.US_ASCII);

        try (FileOutputStream fos = new FileOutputStream(path);
             FileChannel channel = fos.getChannel()) {
            fos.write(new byte[]{(byte) 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0});
            ByteBuffer sizeBuf = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
            sizeBuf.putShort((short) headerBytes.length);
            fos.write(sizeBuf.array());
            fos.write(headerBytes);

            int elem = a.dtype.getByteSize();
            ByteBuffer dataBuf = ByteBuffer.allocate((int) (a.size * elem)).order(ByteOrder.LITTLE_ENDIAN);
            writeElements(a, dataBuf);
            dataBuf.flip();
            channel.write(dataBuf);
        }
    }

    public static NDArray load(String path) throws IOException {
        try (FileInputStream fis = new FileInputStream(path);
             FileChannel channel = fis.getChannel()) {
            byte[] prefix = new byte[10];
            if (fis.read(prefix) != 10) throw new EOFException("truncated npy");
            if (prefix[0] != (byte) 0x93 || prefix[1] != 'N' || prefix[2] != 'U'
                    || prefix[3] != 'M' || prefix[4] != 'P' || prefix[5] != 'Y') {
                throw new IOException("Not a NumPy .npy file: " + path);
            }
            int major = prefix[6] & 0xff;
            int headerLen;
            if (major == 1) {
                headerLen = ByteBuffer.wrap(new byte[]{prefix[8], prefix[9]})
                        .order(ByteOrder.LITTLE_ENDIAN).getShort() & 0xffff;
            } else {
                byte[] rest = new byte[2];
                if (fis.read(rest) != 2) throw new EOFException("truncated npy v2 header len");
                headerLen = ByteBuffer.wrap(new byte[]{prefix[8], prefix[9], rest[0], rest[1]})
                        .order(ByteOrder.LITTLE_ENDIAN).getInt();
            }
            byte[] headerBytes = new byte[headerLen];
            if (fis.read(headerBytes) != headerLen) throw new EOFException("truncated npy header");

            NpyHeader header = NpyHeader.parse(new String(headerBytes, StandardCharsets.US_ASCII));
            long totalSize = header.numel();
            if (totalSize > Integer.MAX_VALUE) {
                throw new IOException("Array too large for Java heap: " + totalSize);
            }
            int bytes = (int) (totalSize * header.dtype.getByteSize());
            ByteBuffer dataBuf = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
            while (dataBuf.hasRemaining()) {
                if (channel.read(dataBuf) < 0) throw new EOFException("truncated npy data");
            }
            dataBuf.flip();
            return readElements(header.dtype, header.shape, dataBuf);
        }
    }

    public static void savez(String path, Map<String, NDArray> arrays) throws IOException {
        try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(path))) {
            for (Map.Entry<String, NDArray> e : arrays.entrySet()) {
                String name = e.getKey().endsWith(".npy") ? e.getKey() : e.getKey() + ".npy";
                zos.putNextEntry(new ZipEntry(name));
                Path tmp = Files.createTempFile("npz-", ".npy");
                try {
                    save(e.getValue(), tmp.toString());
                    Files.copy(tmp, zos);
                } finally {
                    Files.deleteIfExists(tmp);
                }
                zos.closeEntry();
            }
        }
    }

    public static Map<String, NDArray> loadz(String path) throws IOException {
        Map<String, NDArray> out = new LinkedHashMap<>();
        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(path))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.isDirectory()) continue;
                String name = entry.getName();
                if (!name.endsWith(".npy")) continue;
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                byte[] buf = new byte[8192];
                int n;
                while ((n = zis.read(buf)) >= 0) bos.write(buf, 0, n);
                Path tmp = Files.createTempFile("npz-load-", ".npy");
                try {
                    Files.write(tmp, bos.toByteArray());
                    String key = name.endsWith(".npy") ? name.substring(0, name.length() - 4) : name;
                    int slash = key.lastIndexOf('/');
                    if (slash >= 0) key = key.substring(slash + 1);
                    out.put(key, load(tmp.toString()));
                } finally {
                    Files.deleteIfExists(tmp);
                }
            }
        }
        return out;
    }

    private static void writeElements(NDArray a, ByteBuffer buf) {
        DType d = a.dtype;
        for (int i = 0; i < a.size; i++) {
            switch (d) {
                case FLOAT64: buf.putDouble(a.getDouble(i)); break;
                case FLOAT32: buf.putFloat((float) a.getDouble(i)); break;
                case FLOAT16: buf.putShort(floatToHalf((float) a.getDouble(i))); break;
                case INT64: buf.putLong(a.getLong(i)); break;
                case INT32: buf.putInt((int) a.getLong(i)); break;
                case INT16: buf.putShort((short) a.getLong(i)); break;
                case INT8: buf.put((byte) a.getLong(i)); break;
                case UINT8: buf.put((byte) (a.getLong(i) & 0xff)); break;
                case BOOL: buf.put((byte) (a.getLong(i) != 0 ? 1 : 0)); break;
                case COMPLEX128:
                    buf.putDouble(a.getReal(i));
                    buf.putDouble(a.getImag(i));
                    break;
                case COMPLEX64:
                    buf.putFloat((float) a.getReal(i));
                    buf.putFloat((float) a.getImag(i));
                    break;
                default: buf.putDouble(a.getDouble(i)); break;
            }
        }
    }

    private static NDArray readElements(DType d, long[] shape, ByteBuffer buf) {
        NDArray a = new NDArray(d, shape);
        for (int i = 0; i < a.size; i++) {
            switch (d) {
                case FLOAT64: a.setDouble(i, buf.getDouble()); break;
                case FLOAT32: a.setDouble(i, buf.getFloat()); break;
                case FLOAT16: a.setDouble(i, halfToFloat(buf.getShort())); break;
                case INT64: a.setLong(i, buf.getLong()); break;
                case INT32: a.setLong(i, buf.getInt()); break;
                case INT16: a.setLong(i, buf.getShort()); break;
                case INT8: a.setLong(i, buf.get()); break;
                case UINT8: a.setLong(i, buf.get() & 0xff); break;
                case BOOL: a.setLong(i, buf.get() != 0 ? 1 : 0); break;
                case COMPLEX128:
                    a.setComplex(i, buf.getDouble(), buf.getDouble());
                    break;
                case COMPLEX64:
                    a.setComplex(i, buf.getFloat(), buf.getFloat());
                    break;
                default: a.setDouble(i, buf.getDouble()); break;
            }
        }
        return a;
    }

    private static short floatToHalf(float fval) {
        int fbits = Float.floatToIntBits(fval);
        int sign = (fbits >>> 16) & 0x8000;
        int val = (fbits & 0x7fffffff) + 0x1000;
        if (val >= 0x47800000) {
            if ((fbits & 0x7fffffff) >= 0x47800000) {
                if (val < 0x7f800000) return (short) (sign | 0x7c00);
                return (short) (sign | 0x7c00 | ((fbits & 0x007fffff) >>> 13));
            }
            return (short) (sign | 0x7bff);
        }
        if (val >= 0x38800000) return (short) (sign | ((val - 0x38000000) >>> 13));
        if (val < 0x33000000) return (short) sign;
        val = (fbits & 0x7fffffff) >>> 23;
        return (short) (sign | ((((fbits & 0x7fffff) | 0x800000) + (0x800000 >>> (val - 102))) >>> (126 - val)));
    }

    private static float halfToFloat(short hbits) {
        int mant = hbits & 0x03ff;
        int exp = hbits & 0x7c00;
        if (exp == 0x7c00) exp = 0x3fc00;
        else if (exp != 0) {
            exp += 0x1c000;
            if (mant == 0 && exp > 0x1c400)
                return Float.intBitsToFloat((hbits & 0x8000) << 16 | exp << 13);
        } else if (mant != 0) {
            exp = 0x1c400;
            do {
                mant <<= 1;
                exp -= 0x400;
            } while ((mant & 0x400) == 0);
            mant &= 0x3ff;
        }
        return Float.intBitsToFloat((hbits & 0x8000) << 16 | (exp | mant) << 13);
    }

    // ---- Tensor conversion --------------------------------------------------

    public static Tensor toTensor(NDArray a) {
        if (a == null) return null;
        long[] shape = a.shape;
        switch (a.dtype) {
            case FLOAT64: {
                Tensor t = torch.tensor(a.asDoubleArray());
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case FLOAT32:
            case FLOAT16: {
                Tensor t = torch.tensor(a.asFloatArray());
                if (a.dtype == DType.FLOAT16) t = t.to(torch.kHalf());
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case INT64: {
                Tensor t = torch.tensor(a.asLongArray());
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case INT32: {
                Tensor t = torch.tensor(a.asIntArray());
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case INT16: {
                short[] s = new short[(int) a.size];
                for (int i = 0; i < s.length; i++) s[i] = (short) a.getLong(i);
                Tensor t = torch.tensor(s);
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case INT8:
            case UINT8: {
                byte[] b = new byte[(int) a.size];
                for (int i = 0; i < b.length; i++) b[i] = (byte) a.getLong(i);
                Tensor t = torch.tensor(b);
                if (a.dtype == DType.UINT8) t = t.to(torch.kByte());
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case BOOL: {
                boolean[][] as2d = new boolean[1][(int) a.size];
                for (int i = 0; i < a.size; i++) as2d[0][i] = a.getLong(i) != 0;
                return torch.tensor(as2d).reshape(shape.length > 0 ? shape : new long[]{a.size});
            }
            case COMPLEX64: {
                // Interleaved real/imag → complex float tensor via empty + buffer
                long[] sh = shape.length > 0 ? shape : new long[]{a.size};
                Tensor t = torch.empty(sh, new TensorOptions(ScalarType.ComplexFloat), null);
                java.nio.FloatBuffer buf = t.createBuffer();
                for (int i = 0; i < a.size; i++) {
                    buf.put((float) a.getReal(i));
                    buf.put((float) a.getImag(i));
                }
                return t;
            }
            case COMPLEX128: {
                long[] sh = shape.length > 0 ? shape : new long[]{a.size};
                Tensor t = torch.empty(sh, new TensorOptions(ScalarType.ComplexDouble), null);
                java.nio.DoubleBuffer buf = t.createBuffer();
                for (int i = 0; i < a.size; i++) {
                    buf.put(a.getReal(i));
                    buf.put(a.getImag(i));
                }
                return t;
            }
            default:
                throw new IllegalArgumentException("Unsupported dtype: " + a.dtype);
        }
    }

    public static NDArray fromTensor(Tensor t) {
        if (t == null) return null;
        Tensor c = t.contiguous().cpu();
        long[] shape = new long[(int) c.dim()];
        for (int i = 0; i < shape.length; i++) shape[i] = c.sizes().get(i);
        ScalarType st = c.scalar_type();
        DType dtype = DType.fromTorch(st);
        long n = c.numel();
        NDArray a = new NDArray(dtype, shape);
        switch (dtype) {
            case FLOAT64: {
                DoublePointer p = c.data_ptr_double();
                for (int i = 0; i < n; i++) a.setDouble(i, p.get(i));
                break;
            }
            case FLOAT32:
            case FLOAT16: {
                Tensor f = dtype == DType.FLOAT16 ? c.to(torch.kFloat()) : c;
                FloatPointer p = f.data_ptr_float();
                for (int i = 0; i < n; i++) a.setDouble(i, p.get(i));
                break;
            }
            case INT64: {
                LongPointer p = c.data_ptr_long();
                for (int i = 0; i < n; i++) a.setLong(i, p.get(i));
                break;
            }
            case INT32: {
                IntPointer p = c.data_ptr_int();
                for (int i = 0; i < n; i++) a.setLong(i, p.get(i));
                break;
            }
            case INT16: {
                ShortPointer p = c.data_ptr_short();
                for (int i = 0; i < n; i++) a.setLong(i, p.get(i));
                break;
            }
            case INT8: {
                BytePointer p = c.data_ptr_char();
                for (int i = 0; i < n; i++) a.setLong(i, p.get(i));
                break;
            }
            case UINT8: {
                BytePointer p = c.data_ptr_byte();
                for (int i = 0; i < n; i++) a.setLong(i, p.get(i) & 0xff);
                break;
            }
            case BOOL: {
                Tensor flat = c.reshape(n);
                for (int i = 0; i < n; i++) a.setLong(i, flat.get((long) i).item_bool() ? 1 : 0);
                break;
            }
            case COMPLEX64: {
                java.nio.FloatBuffer buf = c.createBuffer();
                for (int i = 0; i < n; i++) {
                    float re = buf.get();
                    float im = buf.get();
                    a.setComplex(i, re, im);
                }
                break;
            }
            case COMPLEX128: {
                java.nio.DoubleBuffer buf = c.createBuffer();
                for (int i = 0; i < n; i++) {
                    double re = buf.get();
                    double im = buf.get();
                    a.setComplex(i, re, im);
                }
                break;
            }
            default:
                throw new IllegalArgumentException("Unsupported dtype: " + dtype);
        }
        return a;
    }
}
