package org.bytedeco.pytorch.data.numpy;

/**
 * Complex-aware constructors and elementwise ops on {@link DType#COMPLEX64}/{@link DType#COMPLEX128}.
 */
public final class NPComplex {
    private NPComplex() {}

    public static NDArray complex(NDArray real, NDArray imag) {
        return complex(real, imag, DType.COMPLEX128);
    }

    public static NDArray complex(NDArray real, NDArray imag, DType dtype) {
        if (!dtype.isComplex()) throw new IllegalArgumentException("dtype must be complex");
        long[] shape = NPArrayUtil.broadcastShapes(real.shape, imag.shape);
        NDArray r = NPShape.broadcast_to(real, shape);
        NDArray i = NPShape.broadcast_to(imag, shape);
        double[] buf = new double[(int) r.size * 2];
        for (int k = 0; k < r.size; k++) {
            buf[k * 2] = r.getDouble(k);
            buf[k * 2 + 1] = i.getDouble(k);
        }
        return new NDArray(buf, dtype, shape);
    }

    public static NDArray complex(double re, double im) {
        return new NDArray(new double[]{re, im}, DType.COMPLEX128);
    }

    public static NDArray real(NDArray a) {
        if (!a.isComplex()) return NP.copy(a);
        NDArray out = new NDArray(DType.FLOAT64, a.shape);
        for (int i = 0; i < a.size; i++) out.setDouble(i, a.getReal(i));
        return out;
    }

    public static NDArray imag(NDArray a) {
        if (!a.isComplex()) return NP.zeros(a.dtype.isComplex() ? DType.FLOAT64 : a.dtype, a.shape);
        NDArray out = new NDArray(DType.FLOAT64, a.shape);
        for (int i = 0; i < a.size; i++) out.setDouble(i, a.getImag(i));
        return out;
    }

    public static NDArray conj(NDArray a) {
        if (!a.isComplex()) return NP.copy(a);
        double[] buf = a.asInterleavedComplex();
        for (int i = 0; i < a.size; i++) buf[i * 2 + 1] = -buf[i * 2 + 1];
        return new NDArray(buf, a.dtype, a.shape.clone());
    }

    public static NDArray angle(NDArray a) {
        NDArray out = new NDArray(DType.FLOAT64, a.shape);
        if (!a.isComplex()) {
            for (int i = 0; i < a.size; i++) out.setDouble(i, Math.atan2(0, a.getDouble(i)));
            return out;
        }
        for (int i = 0; i < a.size; i++) out.setDouble(i, Math.atan2(a.getImag(i), a.getReal(i)));
        return out;
    }

    public static NDArray absolute(NDArray a) {
        NDArray out = new NDArray(DType.FLOAT64, a.shape);
        if (!a.isComplex()) return NPMath.abs(a);
        for (int i = 0; i < a.size; i++) {
            double re = a.getReal(i), im = a.getImag(i);
            out.setDouble(i, Math.hypot(re, im));
        }
        return out;
    }

    public static NDArray add(NDArray a, NDArray b) {
        NDArray ca = asComplex(a), cb = asComplex(b);
        long[] shape = NPArrayUtil.broadcastShapes(ca.shape, cb.shape);
        ca = NPShape.broadcast_to(ca, shape);
        cb = NPShape.broadcast_to(cb, shape);
        double[] out = new double[(int) ca.size * 2];
        for (int i = 0; i < ca.size; i++) {
            out[i * 2] = ca.getReal(i) + cb.getReal(i);
            out[i * 2 + 1] = ca.getImag(i) + cb.getImag(i);
        }
        return new NDArray(out, DType.COMPLEX128, shape);
    }

    public static NDArray subtract(NDArray a, NDArray b) {
        NDArray ca = asComplex(a), cb = asComplex(b);
        long[] shape = NPArrayUtil.broadcastShapes(ca.shape, cb.shape);
        ca = NPShape.broadcast_to(ca, shape);
        cb = NPShape.broadcast_to(cb, shape);
        double[] out = new double[(int) ca.size * 2];
        for (int i = 0; i < ca.size; i++) {
            out[i * 2] = ca.getReal(i) - cb.getReal(i);
            out[i * 2 + 1] = ca.getImag(i) - cb.getImag(i);
        }
        return new NDArray(out, DType.COMPLEX128, shape);
    }

    public static NDArray multiply(NDArray a, NDArray b) {
        NDArray ca = asComplex(a), cb = asComplex(b);
        long[] shape = NPArrayUtil.broadcastShapes(ca.shape, cb.shape);
        ca = NPShape.broadcast_to(ca, shape);
        cb = NPShape.broadcast_to(cb, shape);
        double[] out = new double[(int) ca.size * 2];
        for (int i = 0; i < ca.size; i++) {
            double ar = ca.getReal(i), ai = ca.getImag(i);
            double br = cb.getReal(i), bi = cb.getImag(i);
            out[i * 2] = ar * br - ai * bi;
            out[i * 2 + 1] = ar * bi + ai * br;
        }
        return new NDArray(out, DType.COMPLEX128, shape);
    }

    public static NDArray divide(NDArray a, NDArray b) {
        NDArray ca = asComplex(a), cb = asComplex(b);
        long[] shape = NPArrayUtil.broadcastShapes(ca.shape, cb.shape);
        ca = NPShape.broadcast_to(ca, shape);
        cb = NPShape.broadcast_to(cb, shape);
        double[] out = new double[(int) ca.size * 2];
        for (int i = 0; i < ca.size; i++) {
            double ar = ca.getReal(i), ai = ca.getImag(i);
            double br = cb.getReal(i), bi = cb.getImag(i);
            double den = br * br + bi * bi;
            out[i * 2] = (ar * br + ai * bi) / den;
            out[i * 2 + 1] = (ai * br - ar * bi) / den;
        }
        return new NDArray(out, DType.COMPLEX128, shape);
    }

    public static NDArray exp(NDArray a) {
        NDArray c = asComplex(a);
        double[] out = new double[(int) c.size * 2];
        for (int i = 0; i < c.size; i++) {
            double re = c.getReal(i), im = c.getImag(i);
            double e = Math.exp(re);
            out[i * 2] = e * Math.cos(im);
            out[i * 2 + 1] = e * Math.sin(im);
        }
        return new NDArray(out, DType.COMPLEX128, c.shape.clone());
    }

    public static NDArray log(NDArray a) {
        NDArray c = asComplex(a);
        double[] out = new double[(int) c.size * 2];
        for (int i = 0; i < c.size; i++) {
            double re = c.getReal(i), im = c.getImag(i);
            out[i * 2] = Math.log(Math.hypot(re, im));
            out[i * 2 + 1] = Math.atan2(im, re);
        }
        return new NDArray(out, DType.COMPLEX128, c.shape.clone());
    }

    public static NDArray sqrt(NDArray a) {
        // principal square root
        NDArray c = asComplex(a);
        double[] out = new double[(int) c.size * 2];
        for (int i = 0; i < c.size; i++) {
            double re = c.getReal(i), im = c.getImag(i);
            double r = Math.hypot(re, im);
            double wr = Math.sqrt((r + re) / 2.0);
            double wi = Math.signum(im) * Math.sqrt((r - re) / 2.0);
            if (im == 0 && re < 0) { wr = 0; wi = Math.sqrt(-re); }
            out[i * 2] = wr;
            out[i * 2 + 1] = wi;
        }
        return new NDArray(out, DType.COMPLEX128, c.shape.clone());
    }

    public static NDArray asComplex(NDArray a) {
        if (a.isComplex()) return a;
        return complex(a, NP.zeros(DType.FLOAT64, a.shape), DType.COMPLEX128);
    }

    public static boolean isreal(NDArray a) {
        if (!a.isComplex()) return true;
        for (int i = 0; i < a.size; i++) if (a.getImag(i) != 0) return false;
        return true;
    }

    public static NDArray isrealArray(NDArray a) {
        NDArray out = new NDArray(DType.BOOL, a.shape);
        if (!a.isComplex()) {
            for (int i = 0; i < out.size; i++) out.setLong(i, 1);
            return out;
        }
        for (int i = 0; i < a.size; i++) out.setLong(i, a.getImag(i) == 0 ? 1 : 0);
        return out;
    }

    public static NDArray iscomplex(NDArray a) {
        NDArray out = new NDArray(DType.BOOL, a.shape);
        if (!a.isComplex()) {
            for (int i = 0; i < out.size; i++) out.setLong(i, 0);
            return out;
        }
        for (int i = 0; i < a.size; i++) out.setLong(i, a.getImag(i) != 0 ? 1 : 0);
        return out;
    }
}
