package org.bytedeco.pytorch.data.numpy;

import java.util.Arrays;

/**
 * Shared multi-dimensional indexing, broadcasting and axis-reduction helpers
 * for {@link NP}. Contiguous row-major (C-order) layout only.
 */
final class NPArrayUtil {
    private NPArrayUtil() {}

    static long[] copyShape(long[] shape) {
        return shape == null ? new long[0] : shape.clone();
    }

    static long numel(long[] shape) {
        long n = 1;
        for (long s : shape) n *= s;
        return n;
    }

    /** C-order element strides. */
    static long[] stridesOf(long[] shape) {
        long[] st = new long[shape.length];
        long acc = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            st[i] = acc;
            acc *= shape[i];
        }
        return st;
    }

    static int ravel(int[] idx, long[] strides) {
        int flat = 0;
        for (int i = 0; i < idx.length; i++) flat += (int) (idx[i] * strides[i]);
        return flat;
    }

    /** Normalize axis in [-ndim, ndim). */
    static int normalizeAxis(int axis, int ndim) {
        if (ndim == 0) {
            if (axis == 0 || axis == -1) return 0;
            throw new IllegalArgumentException("axis " + axis + " out of bounds for scalar");
        }
        if (axis < -ndim || axis >= ndim) {
            throw new IllegalArgumentException("axis " + axis + " out of bounds for ndim=" + ndim);
        }
        return axis < 0 ? axis + ndim : axis;
    }

    static int[] normalizeAxes(int[] axes, int ndim) {
        if (axes == null) {
            int[] all = new int[ndim];
            for (int i = 0; i < ndim; i++) all[i] = i;
            return all;
        }
        int[] out = new int[axes.length];
        boolean[] seen = new boolean[Math.max(ndim, 1)];
        for (int i = 0; i < axes.length; i++) {
            int a = normalizeAxis(axes[i], ndim);
            if (ndim > 0 && seen[a]) throw new IllegalArgumentException("duplicate axis " + a);
            if (ndim > 0) seen[a] = true;
            out[i] = a;
        }
        return out;
    }

    static int[] singleAxis(Integer axis, int ndim) {
        if (axis == null) {
            int[] all = new int[ndim];
            for (int i = 0; i < ndim; i++) all[i] = i;
            return all;
        }
        return new int[]{normalizeAxis(axis, ndim)};
    }

    static long[] reduceShape(long[] shape, int[] axes, boolean keepdims) {
        boolean[] red = new boolean[shape.length];
        for (int a : axes) {
            if (a >= 0 && a < red.length) red[a] = true;
        }
        if (keepdims) {
            long[] out = shape.clone();
            for (int a : axes) {
                if (a >= 0 && a < out.length) out[a] = 1;
            }
            return out;
        }
        int n = 0;
        for (int i = 0; i < shape.length; i++) if (!red[i]) n++;
        long[] out = new long[n];
        int k = 0;
        for (int i = 0; i < shape.length; i++) if (!red[i]) out[k++] = shape[i];
        return out;
    }

    /** NumPy-style broadcast shape (right-aligned). */
    static long[] broadcastShapes(long[] a, long[] b) {
        int na = a.length, nb = b.length;
        int n = Math.max(na, nb);
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            long sa = i < n - na ? 1 : a[i - (n - na)];
            long sb = i < n - nb ? 1 : b[i - (n - nb)];
            if (sa == sb) out[i] = sa;
            else if (sa == 1) out[i] = sb;
            else if (sb == 1) out[i] = sa;
            else throw new IllegalArgumentException(
                    "operands could not be broadcast together: "
                            + Arrays.toString(a) + " vs " + Arrays.toString(b));
        }
        return out;
    }

    static int broadcastIndex(int[] outIdx, long[] srcShape, long[] srcStrides) {
        int flat = 0;
        int offset = outIdx.length - srcShape.length;
        for (int i = 0; i < srcShape.length; i++) {
            int oi = outIdx[offset + i];
            int si = srcShape[i] == 1 ? 0 : oi;
            flat += (int) (si * srcStrides[i]);
        }
        return flat;
    }

    static void fillMultiIndex(int flat, long[] shape, long[] strides, int[] idx) {
        for (int d = 0; d < shape.length; d++) {
            idx[d] = (int) ((flat / strides[d]) % shape[d]);
        }
    }

    static NDArray unary(NDArray a, DoubleOp op) {
        return unary(a, a.dtype, op);
    }

    static NDArray unary(NDArray a, DType outDtype, DoubleOp op) {
        NDArray result = new NDArray(outDtype, a.shape);
        for (int i = 0; i < a.size; i++) result.setDouble(i, op.apply(a.getDouble(i)));
        return result;
    }

    static NDArray unaryBool(NDArray a, BoolOp op) {
        NDArray result = new NDArray(DType.BOOL, a.shape);
        for (int i = 0; i < a.size; i++) result.setLong(i, op.apply(a.getDouble(i)) ? 1 : 0);
        return result;
    }

    static NDArray binary(NDArray a, NDArray b, DoubleBinaryOp op) {
        return binary(a, b, promote(a.dtype, b.dtype), op);
    }

    static NDArray binary(NDArray a, NDArray b, DType outDtype, DoubleBinaryOp op) {
        if (Arrays.equals(a.shape, b.shape)) {
            NDArray result = new NDArray(outDtype, a.shape);
            for (int i = 0; i < a.size; i++) result.setDouble(i, op.apply(a.getDouble(i), b.getDouble(i)));
            return result;
        }
        long[] shape = broadcastShapes(a.shape, b.shape);
        long[] aSt = stridesOf(a.shape);
        long[] bSt = stridesOf(b.shape);
        long[] oSt = stridesOf(shape);
        NDArray result = new NDArray(outDtype, shape);
        int n = (int) numel(shape);
        int[] idx = new int[shape.length];
        for (int flat = 0; flat < n; flat++) {
            fillMultiIndex(flat, shape, oSt, idx);
            double av = a.getDouble(broadcastIndex(idx, a.shape, aSt));
            double bv = b.getDouble(broadcastIndex(idx, b.shape, bSt));
            result.setDouble(flat, op.apply(av, bv));
        }
        return result;
    }

    static NDArray binaryBool(NDArray a, NDArray b, BoolBinaryOp op) {
        if (Arrays.equals(a.shape, b.shape)) {
            NDArray out = new NDArray(DType.BOOL, a.shape);
            for (int i = 0; i < a.size; i++) {
                out.setLong(i, op.apply(a.getDouble(i), b.getDouble(i)) ? 1 : 0);
            }
            return out;
        }
        long[] shape = broadcastShapes(a.shape, b.shape);
        long[] aSt = stridesOf(a.shape);
        long[] bSt = stridesOf(b.shape);
        long[] oSt = stridesOf(shape);
        NDArray out = new NDArray(DType.BOOL, shape);
        int n = (int) numel(shape);
        int[] idx = new int[shape.length];
        for (int flat = 0; flat < n; flat++) {
            fillMultiIndex(flat, shape, oSt, idx);
            double av = a.getDouble(broadcastIndex(idx, a.shape, aSt));
            double bv = b.getDouble(broadcastIndex(idx, b.shape, bSt));
            out.setLong(flat, op.apply(av, bv) ? 1 : 0);
        }
        return out;
    }

    static DType promote(DType a, DType b) {
        if (a == DType.FLOAT64 || b == DType.FLOAT64) return DType.FLOAT64;
        if (a == DType.FLOAT32 || b == DType.FLOAT32) return DType.FLOAT32;
        if (a == DType.FLOAT16 || b == DType.FLOAT16) return DType.FLOAT16;
        if (NDArray.isFloatFamily(a)) return a;
        if (NDArray.isFloatFamily(b)) return b;
        if (a == DType.INT64 || b == DType.INT64) return DType.INT64;
        if (a == DType.INT32 || b == DType.INT32) return DType.INT32;
        return a;
    }

    /**
     * Reduce over axes. Empty axes → copy. All axes + !keepdims → scalar NDArray (shape {}).
     */
    static NDArray reduce(NDArray a, Integer axis, boolean keepdims, ReduceOp op, DType outDtype) {
        return reduce(a, singleAxis(axis, a.shape.length), keepdims, op, outDtype);
    }

    static NDArray reduce(NDArray a, int[] axesIn, boolean keepdims, ReduceOp op, DType outDtype) {
        int ndim = a.shape.length;
        if (ndim == 0) {
            NDArray out = new NDArray(outDtype);
            double v = a.getDouble(0);
            out.setDouble(0, op.finish(op.acc(op.init(), v), 1, v));
            return out;
        }
        int[] axes = normalizeAxes(axesIn, ndim);
        if (axes.length == 0) {
            return copyOf(a, outDtype);
        }

        long[] outShape = reduceShape(a.shape, axes, keepdims);
        NDArray out = new NDArray(outDtype, outShape);
        if (out.size == 0) return out;

        boolean[] isRed = new boolean[ndim];
        for (int ax : axes) isRed[ax] = true;

        long[] aSt = stridesOf(a.shape);
        long[] oSt = outShape.length == 0 ? new long[0] : stridesOf(outShape);
        int[] aIdx = new int[ndim];

        long redN = 1;
        for (int ax : axes) redN *= a.shape[ax];

        for (int of = 0; of < out.size; of++) {
            if (keepdims) {
                for (int d = 0; d < ndim; d++) {
                    aIdx[d] = isRed[d] ? 0 : (int) ((of / oSt[d]) % outShape[d]);
                }
            } else if (outShape.length == 0) {
                Arrays.fill(aIdx, 0);
            } else {
                int oPos = 0;
                for (int d = 0; d < ndim; d++) {
                    if (isRed[d]) aIdx[d] = 0;
                    else aIdx[d] = (int) ((of / oSt[oPos]) % outShape[oPos++]);
                }
            }

            double state = op.init();
            int count = 0;
            for (long r = 0; r < redN; r++) {
                long rr = r;
                for (int k = axes.length - 1; k >= 0; k--) {
                    int ax = axes[k];
                    long dim = a.shape[ax];
                    aIdx[ax] = (int) (rr % dim);
                    rr /= dim;
                }
                double v = a.getDouble(ravel(aIdx, aSt));
                state = op.acc(state, v);
                count++;
            }
            out.setDouble(of, op.finish(state, count, Double.NaN));
        }
        return out;
    }

    static NDArray argReduce(NDArray a, Integer axis, boolean keepdims, boolean isMax) {
        if (a.size == 0) {
            NDArray empty = new NDArray(DType.INT64);
            empty.setLong(0, -1);
            return empty;
        }
        if (axis == null) {
            int idx = 0;
            double best = a.getDouble(0);
            for (int i = 1; i < a.size; i++) {
                double v = a.getDouble(i);
                if (isMax ? v > best : v < best) {
                    best = v;
                    idx = i;
                }
            }
            NDArray out = new NDArray(DType.INT64);
            out.setLong(0, idx);
            return out;
        }
        int ax = normalizeAxis(axis, a.shape.length);
        long[] outShape = reduceShape(a.shape, new int[]{ax}, keepdims);
        NDArray out = new NDArray(DType.INT64, outShape);
        long[] aSt = stridesOf(a.shape);
        long[] oSt = outShape.length == 0 ? new long[0] : stridesOf(outShape);
        int[] aIdx = new int[a.shape.length];
        boolean[] isRed = new boolean[a.shape.length];
        isRed[ax] = true;

        for (int of = 0; of < out.size; of++) {
            if (keepdims) {
                for (int d = 0; d < a.shape.length; d++) {
                    aIdx[d] = isRed[d] ? 0 : (int) ((of / oSt[d]) % outShape[d]);
                }
            } else {
                int oPos = 0;
                for (int d = 0; d < a.shape.length; d++) {
                    if (isRed[d]) aIdx[d] = 0;
                    else aIdx[d] = (int) ((of / oSt[oPos]) % outShape[oPos++]);
                }
            }
            int bestI = 0;
            aIdx[ax] = 0;
            double best = a.getDouble(ravel(aIdx, aSt));
            long lim = a.shape[ax];
            for (int i = 1; i < lim; i++) {
                aIdx[ax] = i;
                double v = a.getDouble(ravel(aIdx, aSt));
                if (isMax ? v > best : v < best) {
                    best = v;
                    bestI = i;
                }
            }
            out.setLong(of, bestI);
        }
        return out;
    }

    static NDArray cumulative(NDArray a, Integer axis, DoubleBinaryOp op, DType outDtype) {
        if (axis == null) {
            NDArray flat = new NDArray(outDtype, a.size);
            if (a.size == 0) return flat;
            double acc = a.getDouble(0);
            flat.setDouble(0, acc);
            for (int i = 1; i < a.size; i++) {
                acc = op.apply(acc, a.getDouble(i));
                flat.setDouble(i, acc);
            }
            return flat;
        }
        int ax = normalizeAxis(axis, a.shape.length);
        NDArray out = new NDArray(outDtype, a.shape);
        if (a.size == 0) return out;
        long[] st = stridesOf(a.shape);
        int[] idx = new int[a.shape.length];
        long[] otherShape = new long[Math.max(0, a.shape.length - 1)];
        int k = 0;
        for (int d = 0; d < a.shape.length; d++) if (d != ax) otherShape[k++] = a.shape[d];
        long otherN = numel(otherShape.length == 0 ? new long[]{1} : otherShape);
        if (otherShape.length == 0) otherN = 1;
        long[] otherSt = otherShape.length == 0 ? new long[0] : stridesOf(otherShape);
        long redLen = a.shape[ax];
        for (int o = 0; o < otherN; o++) {
            int p = 0;
            for (int d = 0; d < a.shape.length; d++) {
                if (d == ax) idx[d] = 0;
                else {
                    idx[d] = otherShape.length == 0 ? 0
                            : (int) ((o / otherSt[p]) % otherShape[p]);
                    p++;
                }
            }
            double acc = a.getDouble(ravel(idx, st));
            out.setDouble(ravel(idx, st), acc);
            for (int i = 1; i < redLen; i++) {
                idx[ax] = i;
                acc = op.apply(acc, a.getDouble(ravel(idx, st)));
                out.setDouble(ravel(idx, st), acc);
            }
        }
        return out;
    }

    static NDArray copyOf(NDArray a, DType dtype) {
        NDArray out = new NDArray(dtype, a.shape);
        for (int i = 0; i < a.size; i++) out.setDouble(i, a.getDouble(i));
        return out;
    }

    static NDArray asArray(Object x) {
        if (x == null) throw new IllegalArgumentException("null array-like");
        if (x instanceof NDArray) return (NDArray) x;
        if (x instanceof double[]) return new NDArray((double[]) x);
        if (x instanceof float[]) return new NDArray((float[]) x);
        if (x instanceof long[]) return new NDArray((long[]) x, DType.INT64);
        if (x instanceof int[]) {
            int[] src = (int[]) x;
            long[] longs = new long[src.length];
            for (int i = 0; i < src.length; i++) longs[i] = src[i];
            return new NDArray(longs, DType.INT32);
        }
        if (x instanceof Number) return new NDArray(new double[]{((Number) x).doubleValue()});
        throw new IllegalArgumentException("unsupported array-like: " + x.getClass());
    }

    /** Apply permutation of axes (like NumPy transpose with axes). */
    static NDArray permute(NDArray a, int[] axes) {
        int ndim = a.shape.length;
        int[] ax = axes == null ? null : normalizeAxes(axes, ndim);
        if (ax == null) {
            ax = new int[ndim];
            for (int i = 0; i < ndim; i++) ax[i] = ndim - 1 - i;
        }
        if (ax.length != ndim) throw new IllegalArgumentException("axes must match ndim");
        long[] newShape = new long[ndim];
        for (int i = 0; i < ndim; i++) newShape[i] = a.shape[ax[i]];
        NDArray out = new NDArray(a.dtype, newShape);
        if (a.size == 0) return out;
        long[] aSt = stridesOf(a.shape);
        long[] oSt = stridesOf(newShape);
        int[] oIdx = new int[ndim];
        int[] aIdx = new int[ndim];
        for (int flat = 0; flat < out.size; flat++) {
            fillMultiIndex(flat, newShape, oSt, oIdx);
            for (int i = 0; i < ndim; i++) aIdx[ax[i]] = oIdx[i];
            out.setDouble(flat, a.getDouble(ravel(aIdx, aSt)));
        }
        return out;
    }

    @FunctionalInterface
    interface DoubleOp { double apply(double x); }

    @FunctionalInterface
    interface DoubleBinaryOp { double apply(double a, double b); }

    @FunctionalInterface
    interface BoolOp { boolean apply(double x); }

    @FunctionalInterface
    interface BoolBinaryOp { boolean apply(double a, double b); }

    interface ReduceOp {
        double init();
        double acc(double state, double v);
        double finish(double state, int count, double unused);

        static ReduceOp sum() {
            return new ReduceOp() {
                public double init() { return 0; }
                public double acc(double s, double v) { return s + v; }
                public double finish(double s, int c, double u) { return s; }
            };
        }
        static ReduceOp prod() {
            return new ReduceOp() {
                public double init() { return 1; }
                public double acc(double s, double v) { return s * v; }
                public double finish(double s, int c, double u) { return s; }
            };
        }
        static ReduceOp max() {
            return new ReduceOp() {
                public double init() { return Double.NEGATIVE_INFINITY; }
                public double acc(double s, double v) { return Math.max(s, v); }
                public double finish(double s, int c, double u) { return s; }
            };
        }
        static ReduceOp min() {
            return new ReduceOp() {
                public double init() { return Double.POSITIVE_INFINITY; }
                public double acc(double s, double v) { return Math.min(s, v); }
                public double finish(double s, int c, double u) { return s; }
            };
        }
        static ReduceOp mean() {
            return new ReduceOp() {
                public double init() { return 0; }
                public double acc(double s, double v) { return s + v; }
                public double finish(double s, int c, double u) { return c == 0 ? Double.NaN : s / c; }
            };
        }
        static ReduceOp any() {
            return new ReduceOp() {
                public double init() { return 0; }
                public double acc(double s, double v) { return (s != 0 || v != 0) ? 1 : 0; }
                public double finish(double s, int c, double u) { return s; }
            };
        }
        static ReduceOp all() {
            return new ReduceOp() {
                public double init() { return 1; }
                public double acc(double s, double v) { return (s != 0 && v != 0) ? 1 : 0; }
                public double finish(double s, int c, double u) { return c == 0 ? 1 : s; }
            };
        }
    }
}
