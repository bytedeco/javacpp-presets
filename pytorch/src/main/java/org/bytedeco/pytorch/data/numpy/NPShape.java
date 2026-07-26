package org.bytedeco.pytorch.data.numpy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * NumPy-style shape manipulation, stacking, splitting, broadcasting and tiling.
 */
public final class NPShape {
    private NPShape() {}

    public static NDArray copy(NDArray a) {
        return a.copy();
    }

    public static NDArray astype(NDArray a, DType dtype) {
        if (a.dtype == dtype) return copy(a);
        if (dtype.isComplex() && !a.isComplex()) {
            return NPComplex.complex(a, NP.zeros(DType.FLOAT64, a.shape), dtype);
        }
        if (dtype.isComplex() && a.isComplex()) {
            return new NDArray(a.asInterleavedComplex(), dtype, a.shape.clone());
        }
        if (!dtype.isComplex() && a.isComplex()) {
            NDArray out = new NDArray(dtype, a.shape);
            for (int i = 0; i < a.size; i++) out.setDouble(i, a.getReal(i));
            return out;
        }
        NDArray out = new NDArray(dtype, a.shape);
        for (int i = 0; i < a.size; i++) out.setDouble(i, a.getDouble(i));
        return out;
    }

    public static NDArray reshape(NDArray a, long... newShape) {
        long n = 1;
        int infer = -1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] < 0) {
                if (infer >= 0) throw new IllegalArgumentException("only one -1 allowed");
                infer = i;
            } else {
                n *= newShape[i];
            }
        }
        long[] shape = newShape.clone();
        if (infer >= 0) {
            if (n == 0 || a.size % n != 0) throw new IllegalArgumentException("cannot infer reshape");
            shape[infer] = a.size / n;
            n = a.size;
        }
        if (n != a.size) throw new IllegalArgumentException("reshape size mismatch: " + n + " vs " + a.size);
        NDArray out = new NDArray(a.dtype, shape);
        for (int i = 0; i < a.size; i++) out.setDouble(i, a.getDouble(i));
        return out;
    }

    public static NDArray ravel(NDArray a) { return reshape(a, a.size); }

    public static NDArray flatten(NDArray a) { return ravel(a); }

    public static NDArray transpose(NDArray a) {
        return NPArrayUtil.permute(a, null);
    }

    public static NDArray transpose(NDArray a, int... axes) {
        return NPArrayUtil.permute(a, axes);
    }

    public static NDArray swapaxes(NDArray a, int axis1, int axis2) {
        int ndim = a.shape.length;
        int a1 = NPArrayUtil.normalizeAxis(axis1, ndim);
        int a2 = NPArrayUtil.normalizeAxis(axis2, ndim);
        int[] axes = new int[ndim];
        for (int i = 0; i < ndim; i++) axes[i] = i;
        axes[a1] = a2;
        axes[a2] = a1;
        return NPArrayUtil.permute(a, axes);
    }

    public static NDArray moveaxis(NDArray a, int source, int destination) {
        int ndim = a.shape.length;
        int src = NPArrayUtil.normalizeAxis(source, ndim);
        int dst = NPArrayUtil.normalizeAxis(destination, ndim);
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < ndim; i++) if (i != src) order.add(i);
        order.add(Math.min(dst, order.size()), src);
        int[] axes = new int[ndim];
        for (int i = 0; i < ndim; i++) axes[i] = order.get(i);
        return NPArrayUtil.permute(a, axes);
    }

    public static NDArray expand_dims(NDArray a, int axis) {
        int ndim = a.shape.length + 1;
        int ax = axis < 0 ? axis + ndim : axis;
        if (ax < 0 || ax >= ndim) throw new IllegalArgumentException("axis out of bounds");
        long[] shape = new long[ndim];
        int p = 0;
        for (int i = 0; i < ndim; i++) {
            if (i == ax) shape[i] = 1;
            else shape[i] = a.shape[p++];
        }
        return reshape(a, shape);
    }

    public static NDArray squeeze(NDArray a, Integer axis) {
        if (axis != null) {
            int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
            if (a.shape[ax] != 1) throw new IllegalArgumentException("cannot squeeze axis with size != 1");
            long[] shape = new long[a.shape.length - 1];
            int k = 0;
            for (int i = 0; i < a.shape.length; i++) if (i != ax) shape[k++] = a.shape[i];
            return reshape(a, shape);
        }
        int n = 0;
        for (long s : a.shape) if (s != 1) n++;
        long[] shape = new long[n];
        int k = 0;
        for (long s : a.shape) if (s != 1) shape[k++] = s;
        return reshape(a, shape);
    }

    public static NDArray squeeze(NDArray a) { return squeeze(a, null); }

    public static NDArray flip(NDArray a, Integer axis) {
        if (axis == null) {
            NDArray out = new NDArray(a.dtype, a.shape);
            for (int i = 0; i < a.size; i++) out.setDouble(i, a.getDouble((int) a.size - 1 - i));
            return out;
        }
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        NDArray out = new NDArray(a.dtype, a.shape);
        long[] st = NPArrayUtil.stridesOf(a.shape);
        int[] idx = new int[a.shape.length];
        for (int flat = 0; flat < a.size; flat++) {
            NPArrayUtil.fillMultiIndex(flat, a.shape, st, idx);
            int[] src = idx.clone();
            src[ax] = (int) a.shape[ax] - 1 - idx[ax];
            out.setDouble(flat, a.getDouble(NPArrayUtil.ravel(src, st)));
        }
        return out;
    }

    public static NDArray flip(NDArray a, int axis) { return flip(a, Integer.valueOf(axis)); }

    public static NDArray fliplr(NDArray m) {
        if (m.shape.length < 2) throw new IllegalArgumentException("fliplr requires >=2D");
        return flip(m, 1);
    }

    public static NDArray flipud(NDArray m) {
        if (m.shape.length < 1) throw new IllegalArgumentException("flipud requires >=1D");
        return flip(m, 0);
    }

    public static NDArray rot90(NDArray m, int k, int[] axes) {
        if (m.shape.length < 2) throw new IllegalArgumentException("rot90 requires >=2D");
        int ax0 = axes == null ? 0 : NPArrayUtil.normalizeAxis(axes[0], m.shape.length);
        int ax1 = axes == null ? 1 : NPArrayUtil.normalizeAxis(axes[1], m.shape.length);
        k = ((k % 4) + 4) % 4;
        NDArray out = m;
        // NumPy: rot90 = flip(swapaxes(m, ax0, ax1), ax0) once per 90°
        for (int i = 0; i < k; i++) {
            out = flip(swapaxes(out, ax0, ax1), ax0);
        }
        return out;
    }

    public static NDArray rot90(NDArray m) { return rot90(m, 1, null); }

    public static NDArray rot90(NDArray m, int k) { return rot90(m, k, null); }

    public static NDArray broadcast_to(NDArray a, long... shape) {
        // verify compatibility (right-aligned)
        NPArrayUtil.broadcastShapes(a.shape, shape);
        if (shape.length < a.shape.length) {
            throw new IllegalArgumentException("cannot broadcast to fewer dims");
        }
        long[] aSt = NPArrayUtil.stridesOf(a.shape);
        long[] oSt = NPArrayUtil.stridesOf(shape);
        NDArray out = new NDArray(a.dtype, shape);
        int[] idx = new int[shape.length];
        if (a.isComplex()) {
            for (int flat = 0; flat < out.size; flat++) {
                NPArrayUtil.fillMultiIndex(flat, shape, oSt, idx);
                int src = NPArrayUtil.broadcastIndex(idx, a.shape, aSt);
                out.setComplex(flat, a.getReal(src), a.getImag(src));
            }
        } else {
            for (int flat = 0; flat < out.size; flat++) {
                NPArrayUtil.fillMultiIndex(flat, shape, oSt, idx);
                out.setDouble(flat, a.getDouble(NPArrayUtil.broadcastIndex(idx, a.shape, aSt)));
            }
        }
        return out;
    }

    /** NumPy {@code np.lib.stride_tricks.as_strided}. Strides are in elements. */
    public static NDArray as_strided(NDArray x, long[] shape, long[] strides) {
        return x.asStrided(shape, strides, 0);
    }

    public static NDArray as_strided(NDArray x, long[] shape, long[] strides, long offset) {
        return x.asStrided(shape, strides, offset);
    }

    /**
     * Sliding-window view via {@link #as_strided}.
     * {@code windowShape} gives window size per axis; result rank = x.ndim + window.ndim.
     */
    public static NDArray sliding_window_view(NDArray x, long... windowShape) {
        if (windowShape.length > x.shape.length) {
            throw new IllegalArgumentException("window rank > array rank");
        }
        int n = x.shape.length;
        int w = windowShape.length;
        long[] outShape = new long[n + w];
        long[] outStrides = new long[n + w];
        long[] xSt = x.strides != null ? x.strides.clone() : NPArrayUtil.stridesOf(x.shape);
        // leading dims: free positions
        for (int i = 0; i < n - w; i++) {
            outShape[i] = x.shape[i];
            outStrides[i] = xSt[i];
        }
        for (int i = 0; i < w; i++) {
            int dim = n - w + i;
            if (windowShape[i] > x.shape[dim]) {
                throw new IllegalArgumentException("window larger than dimension");
            }
            outShape[n - w + i] = x.shape[dim] - windowShape[i] + 1;
            outStrides[n - w + i] = xSt[dim];
        }
        for (int i = 0; i < w; i++) {
            int dim = n - w + i;
            outShape[n + i] = windowShape[i];
            outStrides[n + i] = xSt[dim];
        }
        return x.asStrided(outShape, outStrides, 0);
    }

    /**
     * Open mesh grid (broadcast-friendly column/row vectors), NumPy {@code ogrid}.
     * Pass ranges as {@code start:stop:step} encoded via 1D arrays already built, or use
     * {@link #ogridRanges}.
     */
    public static NDArray[] ogrid(NDArray... xi) {
        int n = xi.length;
        NDArray[] out = new NDArray[n];
        for (int d = 0; d < n; d++) {
            long[] shape = new long[n];
            Arrays.fill(shape, 1);
            shape[d] = xi[d].size;
            out[d] = reshape(ravel(xi[d]), shape);
        }
        return out;
    }

    /** Dense mesh grid — equivalent to {@code meshgrid(..., indexing='ij')} then broadcast. */
    public static NDArray[] mgrid(NDArray... xi) {
        NDArray[] open = ogrid(xi);
        long[] shape = new long[xi.length];
        for (int i = 0; i < xi.length; i++) shape[i] = xi[i].size;
        NDArray[] out = new NDArray[xi.length];
        for (int i = 0; i < xi.length; i++) out[i] = broadcast_to(open[i], shape);
        return out;
    }

    /**
     * Build 1D ranges like NumPy slice {@code start:stop:step} for ogrid/mgrid.
     * {@code step == 0} is illegal; if stop&lt;start and step&gt;0 → empty.
     */
    public static NDArray range1d(double start, double stop, double step) {
        return NP.arange(start, stop, step);
    }

    public static NDArray[] broadcast_arrays(NDArray... arrays) {
        if (arrays.length == 0) return new NDArray[0];
        long[] shape = arrays[0].shape;
        for (int i = 1; i < arrays.length; i++) shape = NPArrayUtil.broadcastShapes(shape, arrays[i].shape);
        NDArray[] out = new NDArray[arrays.length];
        for (int i = 0; i < arrays.length; i++) out[i] = broadcast_to(arrays[i], shape);
        return out;
    }

    public static NDArray atleast_1d(NDArray a) {
        if (a.shape.length >= 1) return a;
        return reshape(a, 1);
    }

    public static NDArray atleast_2d(NDArray a) {
        if (a.shape.length >= 2) return a;
        if (a.shape.length == 0) return reshape(a, 1, 1);
        return reshape(a, 1, a.shape[0]);
    }

    public static NDArray atleast_3d(NDArray a) {
        if (a.shape.length >= 3) return a;
        if (a.shape.length == 0) return reshape(a, 1, 1, 1);
        if (a.shape.length == 1) return reshape(a, 1, a.shape[0], 1);
        return reshape(a, a.shape[0], a.shape[1], 1);
    }

    public static NDArray concatenate(NDArray[] arrays, int axis) {
        if (arrays == null || arrays.length == 0) throw new IllegalArgumentException("need arrays");
        int ax = NPArrayUtil.normalizeAxis(axis, arrays[0].shape.length);
        long[] shape = arrays[0].shape.clone();
        long axSum = 0;
        for (NDArray a : arrays) {
            if (a.shape.length != shape.length) throw new IllegalArgumentException("rank mismatch");
            for (int d = 0; d < shape.length; d++) {
                if (d != ax && a.shape[d] != shape[d]) {
                    throw new IllegalArgumentException("shape mismatch at dim " + d);
                }
            }
            axSum += a.shape[ax];
        }
        shape[ax] = axSum;
        NDArray out = new NDArray(arrays[0].dtype, shape);
        long[] oSt = NPArrayUtil.stridesOf(shape);
        long offset = 0;
        for (NDArray a : arrays) {
            long[] aSt = NPArrayUtil.stridesOf(a.shape);
            int[] aIdx = new int[a.shape.length];
            for (int flat = 0; flat < a.size; flat++) {
                NPArrayUtil.fillMultiIndex(flat, a.shape, aSt, aIdx);
                int[] oIdx = aIdx.clone();
                oIdx[ax] = (int) (aIdx[ax] + offset);
                out.setDouble(NPArrayUtil.ravel(oIdx, oSt), a.getDouble(flat));
            }
            offset += a.shape[ax];
        }
        return out;
    }

    public static NDArray concatenate(NDArray a, NDArray b) {
        return concatenate(new NDArray[]{a, b}, 0);
    }

    public static NDArray concatenate(NDArray a, NDArray b, int axis) {
        return concatenate(new NDArray[]{a, b}, axis);
    }

    public static NDArray stack(NDArray[] arrays, int axis) {
        if (arrays == null || arrays.length == 0) throw new IllegalArgumentException("need arrays");
        long[] base = arrays[0].shape;
        for (NDArray a : arrays) {
            if (!Arrays.equals(a.shape, base)) throw new IllegalArgumentException("stack requires same shapes");
        }
        int ndim = base.length + 1;
        int ax = axis < 0 ? axis + ndim : axis;
        if (ax < 0 || ax >= ndim) throw new IllegalArgumentException("axis out of bounds");
        NDArray[] expanded = new NDArray[arrays.length];
        for (int i = 0; i < arrays.length; i++) expanded[i] = expand_dims(arrays[i], ax);
        return concatenate(expanded, ax);
    }

    public static NDArray stack(NDArray a, NDArray b) {
        return stack(new NDArray[]{a, b}, 0);
    }

    public static NDArray hstack(NDArray[] arrays) {
        if (arrays[0].shape.length == 1) return concatenate(arrays, 0);
        return concatenate(arrays, 1);
    }

    public static NDArray vstack(NDArray[] arrays) {
        NDArray[] rows = new NDArray[arrays.length];
        for (int i = 0; i < arrays.length; i++) {
            rows[i] = arrays[i].shape.length == 1 ? reshape(arrays[i], 1, arrays[i].size) : arrays[i];
        }
        return concatenate(rows, 0);
    }

    public static NDArray dstack(NDArray[] arrays) {
        NDArray[] depth = new NDArray[arrays.length];
        for (int i = 0; i < arrays.length; i++) {
            NDArray a = arrays[i];
            if (a.shape.length == 1) depth[i] = reshape(a, 1, a.size, 1);
            else if (a.shape.length == 2) depth[i] = reshape(a, a.shape[0], a.shape[1], 1);
            else depth[i] = a;
        }
        return concatenate(depth, 2);
    }

    public static NDArray[] split(NDArray ary, int sections, int axis) {
        int ax = NPArrayUtil.normalizeAxis(axis, ary.shape.length);
        long len = ary.shape[ax];
        if (len % sections != 0) throw new IllegalArgumentException("array split does not result in equal size");
        long each = len / sections;
        long[] indices = new long[sections - 1];
        for (int i = 0; i < indices.length; i++) indices[i] = each * (i + 1);
        return array_split(ary, indices, axis);
    }

    public static NDArray[] array_split(NDArray ary, long[] indices, int axis) {
        int ax = NPArrayUtil.normalizeAxis(axis, ary.shape.length);
        long len = ary.shape[ax];
        long[] bounds = new long[indices.length + 2];
        bounds[0] = 0;
        System.arraycopy(indices, 0, bounds, 1, indices.length);
        bounds[bounds.length - 1] = len;
        NDArray[] parts = new NDArray[bounds.length - 1];
        long[] st = NPArrayUtil.stridesOf(ary.shape);
        for (int p = 0; p < parts.length; p++) {
            long start = bounds[p], end = bounds[p + 1];
            long[] shape = ary.shape.clone();
            shape[ax] = Math.max(0, end - start);
            NDArray out = new NDArray(ary.dtype, shape);
            if (out.size == 0) { parts[p] = out; continue; }
            long[] oSt = NPArrayUtil.stridesOf(shape);
            int[] idx = new int[ary.shape.length];
            for (int flat = 0; flat < out.size; flat++) {
                NPArrayUtil.fillMultiIndex(flat, shape, oSt, idx);
                int[] src = idx.clone();
                src[ax] = (int) (idx[ax] + start);
                out.setDouble(flat, ary.getDouble(NPArrayUtil.ravel(src, st)));
            }
            parts[p] = out;
        }
        return parts;
    }

    public static NDArray[] array_split(NDArray ary, int sections, int axis) {
        int ax = NPArrayUtil.normalizeAxis(axis, ary.shape.length);
        long len = ary.shape[ax];
        long[] indices = new long[sections - 1];
        long base = len / sections, rem = len % sections;
        long pos = 0;
        for (int i = 0; i < indices.length; i++) {
            pos += base + (i < rem ? 1 : 0);
            indices[i] = pos;
        }
        return array_split(ary, indices, axis);
    }

    public static NDArray[] hsplit(NDArray ary, int sections) {
        if (ary.shape.length == 1) return split(ary, sections, 0);
        return split(ary, sections, 1);
    }

    public static NDArray[] vsplit(NDArray ary, int sections) {
        return split(ary, sections, 0);
    }

    public static NDArray repeat(NDArray a, int repeats, Integer axis) {
        if (axis == null) {
            NDArray out = new NDArray(a.dtype, a.size * repeats);
            int k = 0;
            for (int i = 0; i < a.size; i++) {
                double v = a.getDouble(i);
                for (int r = 0; r < repeats; r++) out.setDouble(k++, v);
            }
            return out;
        }
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        long[] shape = a.shape.clone();
        shape[ax] = a.shape[ax] * repeats;
        NDArray out = new NDArray(a.dtype, shape);
        long[] aSt = NPArrayUtil.stridesOf(a.shape);
        long[] oSt = NPArrayUtil.stridesOf(shape);
        int[] idx = new int[a.shape.length];
        for (int flat = 0; flat < out.size; flat++) {
            NPArrayUtil.fillMultiIndex(flat, shape, oSt, idx);
            int[] src = idx.clone();
            src[ax] = idx[ax] / repeats;
            out.setDouble(flat, a.getDouble(NPArrayUtil.ravel(src, aSt)));
        }
        return out;
    }

    public static NDArray repeat(NDArray a, int repeats) { return repeat(a, repeats, null); }

    public static NDArray tile(NDArray A, long... reps) {
        long[] aShape = A.shape;
        int n = Math.max(aShape.length, reps.length);
        long[] shapeA = new long[n];
        long[] rep = new long[n];
        Arrays.fill(shapeA, 1);
        Arrays.fill(rep, 1);
        System.arraycopy(aShape, 0, shapeA, n - aShape.length, aShape.length);
        System.arraycopy(reps, 0, rep, n - reps.length, reps.length);
        long[] outShape = new long[n];
        for (int i = 0; i < n; i++) outShape[i] = shapeA[i] * rep[i];
        NDArray src = reshape(A, shapeA);
        NDArray out = new NDArray(A.dtype, outShape);
        long[] sSt = NPArrayUtil.stridesOf(shapeA);
        long[] oSt = NPArrayUtil.stridesOf(outShape);
        int[] idx = new int[n];
        for (int flat = 0; flat < out.size; flat++) {
            NPArrayUtil.fillMultiIndex(flat, outShape, oSt, idx);
            int[] srcIdx = new int[n];
            for (int d = 0; d < n; d++) srcIdx[d] = (int) (idx[d] % shapeA[d]);
            out.setDouble(flat, src.getDouble(NPArrayUtil.ravel(srcIdx, sSt)));
        }
        return out;
    }

    public static NDArray roll(NDArray a, int shift, Integer axis) {
        if (axis == null) {
            NDArray out = new NDArray(a.dtype, a.shape);
            int n = (int) a.size;
            if (n == 0) return out;
            int s = ((shift % n) + n) % n;
            for (int i = 0; i < n; i++) out.setDouble(i, a.getDouble((i - s + n) % n));
            return out;
        }
        int ax = NPArrayUtil.normalizeAxis(axis, a.shape.length);
        int n = (int) a.shape[ax];
        if (n == 0) return copy(a);
        int s = ((shift % n) + n) % n;
        NDArray out = new NDArray(a.dtype, a.shape);
        long[] st = NPArrayUtil.stridesOf(a.shape);
        int[] idx = new int[a.shape.length];
        for (int flat = 0; flat < a.size; flat++) {
            NPArrayUtil.fillMultiIndex(flat, a.shape, st, idx);
            int[] src = idx.clone();
            src[ax] = (idx[ax] - s + n) % n;
            out.setDouble(flat, a.getDouble(NPArrayUtil.ravel(src, st)));
        }
        return out;
    }

    public static NDArray roll(NDArray a, int shift) { return roll(a, shift, null); }

    public static NDArray[] meshgrid(NDArray... xi) {
        return meshgrid(true, xi); // NumPy default indexing='xy'
    }

    /**
     * @param indexingXy true → NumPy {@code indexing='xy'} (swap first two output dims);
     *                   false → {@code indexing='ij'}.
     */
    public static NDArray[] meshgrid(boolean indexingXy, NDArray... xi) {
        int n = xi.length;
        if (n == 0) return new NDArray[0];
        long[] outShape = new long[n];
        for (int i = 0; i < n; i++) outShape[i] = xi[i].size;
        if (indexingXy && n >= 2) {
            long t = outShape[0];
            outShape[0] = outShape[1];
            outShape[1] = t;
        }
        NDArray[] out = new NDArray[n];
        for (int d = 0; d < n; d++) {
            long[] bshape = new long[n];
            Arrays.fill(bshape, 1);
            if (indexingXy && n >= 2) {
                if (d == 0) bshape[1] = xi[0].size;
                else if (d == 1) bshape[0] = xi[1].size;
                else bshape[d] = xi[d].size;
            } else {
                bshape[d] = xi[d].size;
            }
            NDArray slim = reshape(ravel(xi[d]), bshape);
            out[d] = broadcast_to(slim, outShape);
        }
        return out;
    }

    public static NDArray diag(NDArray v, int k) {
        if (v.shape.length == 1) {
            int n = (int) v.size + Math.abs(k);
            NDArray out = new NDArray(v.dtype, n, n);
            for (int i = 0; i < v.size; i++) {
                int r = k >= 0 ? i : i - k;
                int c = k >= 0 ? i + k : i;
                if (r >= 0 && r < n && c >= 0 && c < n) out.setDouble(r * n + c, v.getDouble(i));
            }
            return out;
        }
        if (v.shape.length != 2) throw new IllegalArgumentException("diag expects 1D or 2D");
        int rows = (int) v.shape[0], cols = (int) v.shape[1];
        List<Double> vals = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            int j = i + k;
            if (j >= 0 && j < cols) vals.add(v.getDouble(i * cols + j));
        }
        double[] arr = new double[vals.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = vals.get(i);
        return new NDArray(arr);
    }

    public static NDArray diag(NDArray v) { return diag(v, 0); }

    public static NDArray diagonal(NDArray a, int offset, int axis1, int axis2) {
        if (a.shape.length < 2) throw new IllegalArgumentException("diagonal needs >=2D");
        int a1 = NPArrayUtil.normalizeAxis(axis1, a.shape.length);
        int a2 = NPArrayUtil.normalizeAxis(axis2, a.shape.length);
        if (a1 == a2) throw new IllegalArgumentException("axis1 == axis2");
        // For simplicity support last-two / first-two via flatten extract for 2D
        if (a.shape.length == 2) return diag(a, offset);
        throw new UnsupportedOperationException("diagonal for ndim>2 not fully implemented; use 2D");
    }

    public static NDArray ascontiguousarray(NDArray a) { return copy(a); }

    public static NDArray asfortranarray(NDArray a) {
        // Produce F-order layout physically as C-order of transposed dims... for 2D:
        if (a.shape.length != 2) return copy(a);
        NDArray t = transpose(a);
        // data of t is column-major of original when read as (rows,cols) F
        return t; // pragmatic: return transpose view-copy
    }
}
