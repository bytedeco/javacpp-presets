package org.bytedeco.pytorch.data.numpy;

import java.util.Arrays;

/**
 * NumPy-style reductions, cumulative ops, sorting, statistics and selection.
 */
public final class NPReduce {
    private NPReduce() {}

    // ---- full-reduce scalar convenience --------------------------------------

    public static double sum(NDArray a) {
        double s = 0;
        for (int i = 0; i < a.size; i++) s += a.getDouble(i);
        return s;
    }

    public static double mean(NDArray a) {
        return a.size == 0 ? Double.NaN : sum(a) / a.size;
    }

    public static double prod(NDArray a) {
        double p = 1;
        for (int i = 0; i < a.size; i++) p *= a.getDouble(i);
        return p;
    }

    public static double max(NDArray a) {
        if (a.size == 0) return Double.NEGATIVE_INFINITY;
        double m = a.getDouble(0);
        for (int i = 1; i < a.size; i++) m = Math.max(m, a.getDouble(i));
        return m;
    }

    public static double min(NDArray a) {
        if (a.size == 0) return Double.POSITIVE_INFINITY;
        double m = a.getDouble(0);
        for (int i = 1; i < a.size; i++) m = Math.min(m, a.getDouble(i));
        return m;
    }

    public static double var(NDArray a) { return var(a, 0); }

    public static double var(NDArray a, int ddof) {
        if (a.size <= ddof) return Double.NaN;
        double mu = mean(a);
        double acc = 0;
        for (int i = 0; i < a.size; i++) {
            double d = a.getDouble(i) - mu;
            acc += d * d;
        }
        return acc / (a.size - ddof);
    }

    public static double std(NDArray a) { return Math.sqrt(var(a, 0)); }

    public static double std(NDArray a, int ddof) { return Math.sqrt(var(a, ddof)); }

    public static double nansum(NDArray a) {
        double s = 0;
        for (int i = 0; i < a.size; i++) {
            double v = a.getDouble(i);
            if (!Double.isNaN(v)) s += v;
        }
        return s;
    }

    public static double nanmean(NDArray a) {
        double s = 0;
        int n = 0;
        for (int i = 0; i < a.size; i++) {
            double v = a.getDouble(i);
            if (!Double.isNaN(v)) { s += v; n++; }
        }
        return n == 0 ? Double.NaN : s / n;
    }

    public static int argmax(NDArray a) {
        if (a.size == 0) return -1;
        int idx = 0;
        double m = a.getDouble(0);
        for (int i = 1; i < a.size; i++) {
            double v = a.getDouble(i);
            if (v > m) { m = v; idx = i; }
        }
        return idx;
    }

    public static int argmin(NDArray a) {
        if (a.size == 0) return -1;
        int idx = 0;
        double m = a.getDouble(0);
        for (int i = 1; i < a.size; i++) {
            double v = a.getDouble(i);
            if (v < m) { m = v; idx = i; }
        }
        return idx;
    }

    public static boolean any(NDArray a) {
        for (int i = 0; i < a.size; i++) if (a.getDouble(i) != 0) return true;
        return false;
    }

    public static boolean all(NDArray a) {
        for (int i = 0; i < a.size; i++) if (a.getDouble(i) == 0) return false;
        return true;
    }

    public static double median(NDArray a) {
        if (a.size == 0) return Double.NaN;
        double[] v = a.asDoubleArray().clone();
        Arrays.sort(v);
        int n = v.length;
        if ((n & 1) == 1) return v[n / 2];
        return 0.5 * (v[n / 2 - 1] + v[n / 2]);
    }

    public static double percentile(NDArray a, double q) {
        if (a.size == 0) return Double.NaN;
        if (q < 0 || q > 100) throw new IllegalArgumentException("percentile q in [0,100]");
        double[] v = a.asDoubleArray().clone();
        Arrays.sort(v);
        if (v.length == 1) return v[0];
        double pos = q / 100.0 * (v.length - 1);
        int lo = (int) Math.floor(pos);
        int hi = (int) Math.ceil(pos);
        if (lo == hi) return v[lo];
        double w = pos - lo;
        return v[lo] * (1 - w) + v[hi] * w;
    }

    // ---- axis reductions → NDArray ------------------------------------------

    public static NDArray sum(NDArray a, Integer axis, boolean keepdims) {
        return NPArrayUtil.reduce(a, axis, keepdims, NPArrayUtil.ReduceOp.sum(),
                NDArray.isFloatFamily(a.dtype) ? a.dtype : DType.FLOAT64);
    }

    public static NDArray mean(NDArray a, Integer axis, boolean keepdims) {
        return NPArrayUtil.reduce(a, axis, keepdims, NPArrayUtil.ReduceOp.mean(), DType.FLOAT64);
    }

    public static NDArray prod(NDArray a, Integer axis, boolean keepdims) {
        return NPArrayUtil.reduce(a, axis, keepdims, NPArrayUtil.ReduceOp.prod(),
                NDArray.isFloatFamily(a.dtype) ? a.dtype : DType.FLOAT64);
    }

    public static NDArray max(NDArray a, Integer axis, boolean keepdims) {
        return NPArrayUtil.reduce(a, axis, keepdims, NPArrayUtil.ReduceOp.max(), a.dtype);
    }

    public static NDArray min(NDArray a, Integer axis, boolean keepdims) {
        return NPArrayUtil.reduce(a, axis, keepdims, NPArrayUtil.ReduceOp.min(), a.dtype);
    }

    public static NDArray any(NDArray a, Integer axis, boolean keepdims) {
        return NPArrayUtil.reduce(a, axis, keepdims, NPArrayUtil.ReduceOp.any(), DType.BOOL);
    }

    public static NDArray all(NDArray a, Integer axis, boolean keepdims) {
        return NPArrayUtil.reduce(a, axis, keepdims, NPArrayUtil.ReduceOp.all(), DType.BOOL);
    }

    public static NDArray argmax(NDArray a, Integer axis, boolean keepdims) {
        return NPArrayUtil.argReduce(a, axis, keepdims, true);
    }

    public static NDArray argmin(NDArray a, Integer axis, boolean keepdims) {
        return NPArrayUtil.argReduce(a, axis, keepdims, false);
    }

    public static NDArray var(NDArray a, Integer axis, boolean keepdims, int ddof) {
        NDArray mu = mean(a, axis, true);
        NDArray diff = NPMath.subtract(a, broadcastLike(mu, a.shape));
        NDArray sq = NPMath.square(diff);
        NDArray s = sum(sq, axis, keepdims);
        long red = redCount(a.shape, axis);
        double denom = red - ddof;
        if (denom <= 0) return NPArrayUtil.unary(s, x -> Double.NaN);
        return NPArrayUtil.unary(s, x -> x / denom);
    }

    public static NDArray var(NDArray a, Integer axis, boolean keepdims) {
        return var(a, axis, keepdims, 0);
    }

    public static NDArray std(NDArray a, Integer axis, boolean keepdims, int ddof) {
        return NPMath.sqrt(var(a, axis, keepdims, ddof));
    }

    public static NDArray std(NDArray a, Integer axis, boolean keepdims) {
        return std(a, axis, keepdims, 0);
    }

    public static NDArray cumsum(NDArray a, Integer axis) {
        return NPArrayUtil.cumulative(a, axis, Double::sum,
                NDArray.isFloatFamily(a.dtype) ? a.dtype : DType.FLOAT64);
    }

    public static NDArray cumprod(NDArray a, Integer axis) {
        return NPArrayUtil.cumulative(a, axis, (x, y) -> x * y,
                NDArray.isFloatFamily(a.dtype) ? a.dtype : DType.FLOAT64);
    }

    public static NDArray median(NDArray a, Integer axis, boolean keepdims) {
        if (axis == null) {
            NDArray out = new NDArray(DType.FLOAT64);
            out.setDouble(0, median(a));
            return out;
        }
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        long[] outShape = NPArrayUtil.reduceShape(a.shape, new int[]{ax}, keepdims);
        NDArray out = new NDArray(DType.FLOAT64, outShape);
        long[] aSt = NPArrayUtil.stridesOf(a.shape);
        long[] oSt = outShape.length == 0 ? new long[0] : NPArrayUtil.stridesOf(outShape);
        int[] aIdx = new int[a.shape.length];
        boolean[] isRed = new boolean[a.shape.length];
        isRed[ax] = true;
        int n = (int) a.shape[ax];
        double[] buf = new double[n];
        for (int of = 0; of < out.size; of++) {
            scatterIndex(aIdx, isRed, a.shape, outShape, oSt, of, keepdims);
            for (int i = 0; i < n; i++) {
                aIdx[ax] = i;
                buf[i] = a.getDouble(NPArrayUtil.ravel(aIdx, aSt));
            }
            Arrays.sort(buf);
            double med = (n & 1) == 1 ? buf[n / 2] : 0.5 * (buf[n / 2 - 1] + buf[n / 2]);
            out.setDouble(of, med);
        }
        return out;
    }

    public static NDArray percentile(NDArray a, double q, Integer axis, boolean keepdims) {
        if (axis == null) {
            NDArray out = new NDArray(DType.FLOAT64);
            out.setDouble(0, percentile(a, q));
            return out;
        }
        if (q < 0 || q > 100) throw new IllegalArgumentException("percentile q in [0,100]");
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        long[] outShape = NPArrayUtil.reduceShape(a.shape, new int[]{ax}, keepdims);
        NDArray out = new NDArray(DType.FLOAT64, outShape);
        long[] aSt = NPArrayUtil.stridesOf(a.shape);
        long[] oSt = outShape.length == 0 ? new long[0] : NPArrayUtil.stridesOf(outShape);
        int[] aIdx = new int[a.shape.length];
        boolean[] isRed = new boolean[a.shape.length];
        isRed[ax] = true;
        int n = (int) a.shape[ax];
        double[] buf = new double[n];
        for (int of = 0; of < out.size; of++) {
            scatterIndex(aIdx, isRed, a.shape, outShape, oSt, of, keepdims);
            for (int i = 0; i < n; i++) {
                aIdx[ax] = i;
                buf[i] = a.getDouble(NPArrayUtil.ravel(aIdx, aSt));
            }
            Arrays.sort(buf);
            if (n == 1) { out.setDouble(of, buf[0]); continue; }
            double pos = q / 100.0 * (n - 1);
            int lo = (int) Math.floor(pos);
            int hi = (int) Math.ceil(pos);
            double w = pos - lo;
            out.setDouble(of, lo == hi ? buf[lo] : buf[lo] * (1 - w) + buf[hi] * w);
        }
        return out;
    }

    // ---- sort / partition / unique ------------------------------------------

    public static NDArray sort(NDArray a, Integer axis) {
        if (axis == null) {
            double[] v = a.asDoubleArray().clone();
            Arrays.sort(v);
            return new NDArray(v, a.size);
        }
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        NDArray out = NP.copy(a);
        long[] st = NPArrayUtil.stridesOf(a.shape);
        int n = (int) a.shape[ax];
        double[] buf = new double[n];
        int[] idx = new int[a.shape.length];
        long otherN = a.size / n;
        long[] otherShape = new long[Math.max(0, a.shape.length - 1)];
        int k = 0;
        for (int d = 0; d < a.shape.length; d++) if (d != ax) otherShape[k++] = a.shape[d];
        long[] otherSt = otherShape.length == 0 ? new long[0] : NPArrayUtil.stridesOf(otherShape);
        for (int o = 0; o < otherN; o++) {
            int p = 0;
            for (int d = 0; d < a.shape.length; d++) {
                if (d == ax) idx[d] = 0;
                else {
                    idx[d] = otherShape.length == 0 ? 0 : (int) ((o / otherSt[p]) % otherShape[p]);
                    p++;
                }
            }
            for (int i = 0; i < n; i++) {
                idx[ax] = i;
                buf[i] = a.getDouble(NPArrayUtil.ravel(idx, st));
            }
            Arrays.sort(buf);
            for (int i = 0; i < n; i++) {
                idx[ax] = i;
                out.setDouble(NPArrayUtil.ravel(idx, st), buf[i]);
            }
        }
        return out;
    }

    public static NDArray argsort(NDArray a, Integer axis) {
        if (axis == null) {
            Integer[] order = new Integer[(int) a.size];
            for (int i = 0; i < order.length; i++) order[i] = i;
            Arrays.sort(order, (i, j) -> Double.compare(a.getDouble(i), a.getDouble(j)));
            long[] ids = new long[order.length];
            for (int i = 0; i < order.length; i++) ids[i] = order[i];
            return new NDArray(ids, DType.INT64, a.size);
        }
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        NDArray out = new NDArray(DType.INT64, a.shape);
        long[] st = NPArrayUtil.stridesOf(a.shape);
        int n = (int) a.shape[ax];
        Integer[] order = new Integer[n];
        double[] buf = new double[n];
        int[] idx = new int[a.shape.length];
        long otherN = a.size / Math.max(n, 1);
        long[] otherShape = new long[Math.max(0, a.shape.length - 1)];
        int k = 0;
        for (int d = 0; d < a.shape.length; d++) if (d != ax) otherShape[k++] = a.shape[d];
        long[] otherSt = otherShape.length == 0 ? new long[0] : NPArrayUtil.stridesOf(otherShape);
        for (int o = 0; o < otherN; o++) {
            int p = 0;
            for (int d = 0; d < a.shape.length; d++) {
                if (d == ax) idx[d] = 0;
                else {
                    idx[d] = otherShape.length == 0 ? 0 : (int) ((o / otherSt[p]) % otherShape[p]);
                    p++;
                }
            }
            for (int i = 0; i < n; i++) {
                idx[ax] = i;
                buf[i] = a.getDouble(NPArrayUtil.ravel(idx, st));
                order[i] = i;
            }
            final double[] b = buf;
            Arrays.sort(order, (i, j) -> Double.compare(b[i], b[j]));
            for (int i = 0; i < n; i++) {
                idx[ax] = i;
                out.setLong(NPArrayUtil.ravel(idx, st), order[i]);
            }
        }
        return out;
    }

    public static NDArray partition(NDArray a, int kth, Integer axis) {
        if (axis == null) {
            double[] v = a.asDoubleArray().clone();
            int kt = normalizeKth(kth, v.length);
            quickselect(v, 0, v.length - 1, kt);
            // ensure left of kth <= v[kth] (quickselect guarantees)
            return new NDArray(v, a.size);
        }
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        NDArray out = NP.copy(a);
        long[] st = NPArrayUtil.stridesOf(a.shape);
        int n = (int) a.shape[ax];
        int kt = normalizeKth(kth, n);
        double[] buf = new double[n];
        int[] idx = new int[a.shape.length];
        long otherN = a.size / Math.max(n, 1);
        long[] otherShape = new long[Math.max(0, a.shape.length - 1)];
        int k = 0;
        for (int d = 0; d < a.shape.length; d++) if (d != ax) otherShape[k++] = a.shape[d];
        long[] otherSt = otherShape.length == 0 ? new long[0] : NPArrayUtil.stridesOf(otherShape);
        for (int o = 0; o < otherN; o++) {
            int p = 0;
            for (int d = 0; d < a.shape.length; d++) {
                if (d == ax) idx[d] = 0;
                else {
                    idx[d] = otherShape.length == 0 ? 0 : (int) ((o / otherSt[p]) % otherShape[p]);
                    p++;
                }
            }
            for (int i = 0; i < n; i++) {
                idx[ax] = i;
                buf[i] = a.getDouble(NPArrayUtil.ravel(idx, st));
            }
            quickselect(buf, 0, n - 1, kt);
            for (int i = 0; i < n; i++) {
                idx[ax] = i;
                out.setDouble(NPArrayUtil.ravel(idx, st), buf[i]);
            }
        }
        return out;
    }

    public static NDArray argpartition(NDArray a, int kth, Integer axis) {
        if (axis == null) {
            int n = (int) a.size;
            int kt = normalizeKth(kth, n);
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) order[i] = i;
            double[] vals = a.asDoubleArray();
            quickselectIdx(order, vals, 0, n - 1, kt);
            long[] ids = new long[n];
            for (int i = 0; i < n; i++) ids[i] = order[i];
            return new NDArray(ids, DType.INT64, a.size);
        }
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        NDArray out = new NDArray(DType.INT64, a.shape);
        long[] st = NPArrayUtil.stridesOf(a.shape);
        int n = (int) a.shape[ax];
        int kt = normalizeKth(kth, n);
        Integer[] order = new Integer[n];
        double[] buf = new double[n];
        int[] idx = new int[a.shape.length];
        long otherN = a.size / Math.max(n, 1);
        long[] otherShape = new long[Math.max(0, a.shape.length - 1)];
        int k = 0;
        for (int d = 0; d < a.shape.length; d++) if (d != ax) otherShape[k++] = a.shape[d];
        long[] otherSt = otherShape.length == 0 ? new long[0] : NPArrayUtil.stridesOf(otherShape);
        for (int o = 0; o < otherN; o++) {
            int p = 0;
            for (int d = 0; d < a.shape.length; d++) {
                if (d == ax) idx[d] = 0;
                else {
                    idx[d] = otherShape.length == 0 ? 0 : (int) ((o / otherSt[p]) % otherShape[p]);
                    p++;
                }
            }
            for (int i = 0; i < n; i++) {
                idx[ax] = i;
                buf[i] = a.getDouble(NPArrayUtil.ravel(idx, st));
                order[i] = i;
            }
            quickselectIdx(order, buf, 0, n - 1, kt);
            for (int i = 0; i < n; i++) {
                idx[ax] = i;
                out.setLong(NPArrayUtil.ravel(idx, st), order[i]);
            }
        }
        return out;
    }

    private static int normalizeKth(int kth, int n) {
        if (n == 0) return 0;
        if (kth < 0) kth += n;
        if (kth < 0 || kth >= n) throw new IllegalArgumentException("kth out of bounds: " + kth);
        return kth;
    }

    /** In-place quickselect: after return, {@code a[k]} is the k-th smallest; left side <=, right >=. */
    private static void quickselect(double[] a, int left, int right, int k) {
        while (left < right) {
            int pivotIndex = partitionRange(a, left, right);
            if (k == pivotIndex) return;
            if (k < pivotIndex) right = pivotIndex - 1;
            else left = pivotIndex + 1;
        }
    }

    private static int partitionRange(double[] a, int left, int right) {
        int mid = left + (right - left) / 2;
        double pivot = a[mid];
        swap(a, mid, right);
        int store = left;
        for (int i = left; i < right; i++) {
            if (a[i] < pivot) swap(a, store++, i);
        }
        swap(a, store, right);
        return store;
    }

    private static void swap(double[] a, int i, int j) {
        double t = a[i]; a[i] = a[j]; a[j] = t;
    }

    private static void quickselectIdx(Integer[] order, double[] vals, int left, int right, int k) {
        while (left < right) {
            int pivotIndex = partitionIdx(order, vals, left, right);
            if (k == pivotIndex) return;
            if (k < pivotIndex) right = pivotIndex - 1;
            else left = pivotIndex + 1;
        }
    }

    private static int partitionIdx(Integer[] order, double[] vals, int left, int right) {
        int mid = left + (right - left) / 2;
        double pivot = vals[order[mid]];
        swapIdx(order, mid, right);
        int store = left;
        for (int i = left; i < right; i++) {
            if (vals[order[i]] < pivot) swapIdx(order, store++, i);
        }
        swapIdx(order, store, right);
        return store;
    }

    private static void swapIdx(Integer[] a, int i, int j) {
        Integer t = a[i]; a[i] = a[j]; a[j] = t;
    }

    public static NDArray unique(NDArray a) {
        double[] v = a.asDoubleArray().clone();
        Arrays.sort(v);
        int n = 0;
        for (int i = 0; i < v.length; i++) {
            if (i == 0 || v[i] != v[i - 1]) v[n++] = v[i];
        }
        return new NDArray(Arrays.copyOf(v, n));
    }

    public static NDArray searchsorted(NDArray a, NDArray v) {
        // a must be 1D sorted
        NDArray out = new NDArray(DType.INT64, v.shape);
        for (int i = 0; i < v.size; i++) {
            double x = v.getDouble(i);
            int lo = 0, hi = (int) a.size;
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                if (a.getDouble(mid) < x) lo = mid + 1;
                else hi = mid;
            }
            out.setLong(i, lo);
        }
        return out;
    }

    public static NDArray bincount(NDArray x, NDArray weights, int minlength) {
        int max = minlength - 1;
        for (int i = 0; i < x.size; i++) {
            int v = (int) x.getLong(i);
            if (v < 0) throw new IllegalArgumentException("bincount requires non-negative");
            if (v > max) max = v;
        }
        NDArray out = new NDArray(weights == null ? DType.INT64 : DType.FLOAT64, max + 1);
        for (int i = 0; i < x.size; i++) {
            int bin = (int) x.getLong(i);
            if (weights == null) out.setLong(bin, out.getLong(bin) + 1);
            else out.setDouble(bin, out.getDouble(bin) + weights.getDouble(i));
        }
        return out;
    }

    public static NDArray bincount(NDArray x) { return bincount(x, null, 0); }

    public static NDArray[] histogram(NDArray a, int bins) {
        if (bins <= 0) throw new IllegalArgumentException("bins must be > 0");
        double lo = min(a), hi = max(a);
        if (lo == hi) { hi = lo + 1; }
        double width = (hi - lo) / bins;
        NDArray counts = NP.zeros(DType.INT64, bins);
        NDArray edges = new NDArray(DType.FLOAT64, bins + 1);
        for (int i = 0; i <= bins; i++) edges.setDouble(i, lo + i * width);
        for (int i = 0; i < a.size; i++) {
            double v = a.getDouble(i);
            int b = (int) ((v - lo) / width);
            if (b < 0) b = 0;
            if (b >= bins) b = bins - 1;
            counts.setLong(b, counts.getLong(b) + 1);
        }
        return new NDArray[]{counts, edges};
    }

    public static NDArray digitize(NDArray x, NDArray bins) {
        return searchsorted(bins, x);
    }

    public static NDArray extract(NDArray condition, NDArray arr) {
        int n = 0;
        for (int i = 0; i < condition.size; i++) if (condition.getDouble(i) != 0) n++;
        NDArray out = new NDArray(arr.dtype, n);
        int k = 0;
        for (int i = 0; i < condition.size; i++) {
            if (condition.getDouble(i) != 0) out.setDouble(k++, arr.getDouble(i));
        }
        return out;
    }

    public static NDArray[] nonzero(NDArray a) {
        int n = 0;
        for (int i = 0; i < a.size; i++) if (a.getDouble(i) != 0) n++;
        int ndim = Math.max(a.shape.length, 1);
        if (a.shape.length == 0) ndim = 1;
        NDArray[] coords = new NDArray[a.shape.length == 0 ? 1 : a.shape.length];
        for (int d = 0; d < coords.length; d++) coords[d] = new NDArray(DType.INT64, n);
        if (a.shape.length == 0) {
            if (a.getDouble(0) != 0) coords[0].setLong(0, 0);
            return coords;
        }
        long[] st = NPArrayUtil.stridesOf(a.shape);
        int[] idx = new int[a.shape.length];
        int k = 0;
        for (int flat = 0; flat < a.size; flat++) {
            if (a.getDouble(flat) == 0) continue;
            NPArrayUtil.fillMultiIndex(flat, a.shape, st, idx);
            for (int d = 0; d < a.shape.length; d++) coords[d].setLong(k, idx[d]);
            k++;
        }
        return coords;
    }

    public static NDArray[] where(NDArray cond) { return nonzero(cond); }

    public static NDArray diff(NDArray a, int n, Integer axis) {
        if (n < 0) throw new IllegalArgumentException("n must be >= 0");
        if (n == 0) return NP.copy(a);
        int ax = axis == null ? a.shape.length - 1 : NPArrayUtil.normalizeAxis(axis, a.shape.length);
        NDArray cur = a;
        for (int step = 0; step < n; step++) {
            if (cur.shape[ax] < 2) {
                long[] sh = cur.shape.clone();
                sh[ax] = 0;
                return new NDArray(cur.dtype, sh);
            }
            long[] newShape = cur.shape.clone();
            newShape[ax] = cur.shape[ax] - 1;
            NDArray out = new NDArray(cur.dtype, newShape);
            long[] cSt = NPArrayUtil.stridesOf(cur.shape);
            long[] oSt = NPArrayUtil.stridesOf(newShape);
            int[] idx = new int[cur.shape.length];
            for (int flat = 0; flat < out.size; flat++) {
                NPArrayUtil.fillMultiIndex(flat, newShape, oSt, idx);
                int[] idx1 = idx.clone();
                idx1[ax] = idx[ax] + 1;
                out.setDouble(flat, cur.getDouble(NPArrayUtil.ravel(idx1, cSt))
                        - cur.getDouble(NPArrayUtil.ravel(idx, cSt)));
            }
            cur = out;
        }
        return cur;
    }

    public static NDArray diff(NDArray a) { return diff(a, 1, null); }

    public static NDArray ediff1d(NDArray ary) {
        NDArray flat = NPShape.ravel(ary);
        return diff(flat, 1, 0);
    }

    public static NDArray setdiff1d(NDArray x, NDArray y) {
        NDArray ux = unique(x);
        NDArray uy = unique(y);
        int n = 0;
        double[] buf = new double[(int) ux.size];
        for (int i = 0; i < ux.size; i++) {
            double v = ux.getDouble(i);
            boolean found = false;
            for (int j = 0; j < uy.size; j++) if (uy.getDouble(j) == v) { found = true; break; }
            if (!found) buf[n++] = v;
        }
        return new NDArray(Arrays.copyOf(buf, n));
    }

    public static NDArray interp(NDArray x, NDArray xp, NDArray fp) {
        NDArray out = new NDArray(DType.FLOAT64, x.shape);
        for (int i = 0; i < x.size; i++) {
            double xi = x.getDouble(i);
            if (xi <= xp.getDouble(0)) { out.setDouble(i, fp.getDouble(0)); continue; }
            int last = (int) xp.size - 1;
            if (xi >= xp.getDouble(last)) { out.setDouble(i, fp.getDouble(last)); continue; }
            int lo = 0, hi = last;
            while (hi - lo > 1) {
                int mid = (lo + hi) >>> 1;
                if (xp.getDouble(mid) <= xi) lo = mid; else hi = mid;
            }
            double x0 = xp.getDouble(lo), x1 = xp.getDouble(hi);
            double y0 = fp.getDouble(lo), y1 = fp.getDouble(hi);
            double t = (xi - x0) / (x1 - x0);
            out.setDouble(i, y0 + t * (y1 - y0));
        }
        return out;
    }

    public static NDArray cov(NDArray m, boolean rowvar) {
        NDArray data = rowvar ? m : NPShape.transpose(m);
        if (data.shape.length != 2) throw new IllegalArgumentException("cov expects 2D");
        int vars = (int) data.shape[0];
        int obs = (int) data.shape[1];
        NDArray means = mean(data, 1, true);
        NDArray centered = NPMath.subtract(data, broadcastLike(means, data.shape));
        NDArray out = new NDArray(DType.FLOAT64, vars, vars);
        double denom = Math.max(obs - 1, 1);
        for (int i = 0; i < vars; i++) {
            for (int j = 0; j < vars; j++) {
                double s = 0;
                for (int k = 0; k < obs; k++) {
                    s += centered.getDouble(i * obs + k) * centered.getDouble(j * obs + k);
                }
                out.setDouble(i * vars + j, s / denom);
            }
        }
        return out;
    }

    public static NDArray cov(NDArray m) { return cov(m, true); }

    public static NDArray corrcoef(NDArray x, boolean rowvar) {
        NDArray c = cov(x, rowvar);
        int n = (int) c.shape[0];
        NDArray out = new NDArray(DType.FLOAT64, n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double d = Math.sqrt(c.getDouble(i * n + i) * c.getDouble(j * n + j));
                out.setDouble(i * n + j, d == 0 ? Double.NaN : c.getDouble(i * n + j) / d);
            }
        }
        return out;
    }

    public static NDArray corrcoef(NDArray x) { return corrcoef(x, true); }

    public static NDArray convolve(NDArray a, NDArray v, String mode) {
        NDArray fa = NPShape.ravel(a);
        NDArray fv = NPShape.ravel(v);
        int n = (int) fa.size, m = (int) fv.size;
        int full = n + m - 1;
        double[] acc = new double[full];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                acc[i + j] += fa.getDouble(i) * fv.getDouble(j);
            }
        }
        if (mode == null) mode = "full";
        switch (mode) {
            case "full": return new NDArray(acc);
            case "same": {
                int outLen = n;
                int start = (full - outLen) / 2;
                double[] o = Arrays.copyOfRange(acc, start, start + outLen);
                return new NDArray(o);
            }
            case "valid": {
                int outLen = Math.max(n - m + 1, m - n + 1);
                int start = Math.min(n, m) - 1;
                double[] o = Arrays.copyOfRange(acc, start, start + outLen);
                return new NDArray(o);
            }
            default: throw new IllegalArgumentException("mode: " + mode);
        }
    }

    public static NDArray correlate(NDArray a, NDArray v, String mode) {
        NDArray vr = NPShape.flip(NPShape.ravel(v), 0);
        return convolve(a, vr, mode == null ? "valid" : mode);
    }

    // ---- helpers ------------------------------------------------------------

    private static void scatterIndex(int[] aIdx, boolean[] isRed, long[] aShape,
                                     long[] outShape, long[] oSt, int of, boolean keepdims) {
        if (keepdims) {
            for (int d = 0; d < aShape.length; d++) {
                aIdx[d] = isRed[d] ? 0 : (int) ((of / oSt[d]) % outShape[d]);
            }
        } else if (outShape.length == 0) {
            Arrays.fill(aIdx, 0);
        } else {
            int oPos = 0;
            for (int d = 0; d < aShape.length; d++) {
                if (isRed[d]) aIdx[d] = 0;
                else aIdx[d] = (int) ((of / oSt[oPos]) % outShape[oPos++]);
            }
        }
    }

    private static long redCount(long[] shape, Integer axis) {
        if (axis == null) {
            long n = 1;
            for (long s : shape) n *= s;
            return n;
        }
        int ax = NPArrayUtil.normalizeAxis(axis, shape.length);
        return shape[ax];
    }

    private static NDArray broadcastLike(NDArray src, long[] targetShape) {
        return NPShape.broadcast_to(src, targetShape);
    }
}
