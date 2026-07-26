package org.bytedeco.pytorch.data.numpy;

/**
 * NumPy-style polynomial helpers: polyfit / polyval / roots / poly1d companion ops.
 */
public final class NPPoly {
    private NPPoly() {}

    /**
     * Least-squares polynomial fit of degree {@code deg}:
     * {@code y ≈ p[0]*x^deg + ... + p[deg]}.
     */
    public static NDArray polyfit(NDArray x, NDArray y, int deg) {
        if (deg < 0) throw new IllegalArgumentException("deg must be >= 0");
        NDArray xf = NPShape.ravel(x);
        NDArray yf = NPShape.ravel(y);
        if (xf.size != yf.size) throw new IllegalArgumentException("x/y size mismatch");
        int n = (int) xf.size;
        int m = deg + 1;
        if (n < m) throw new IllegalArgumentException("need at least deg+1 points");

        // Vandermonde (n x m) with decreasing powers
        NDArray V = NPLinalg.vander(xf, m, false);
        NDArray[] sol = NPLinalg.lstsq(V, yf);
        return NPShape.ravel(sol[0]);
    }

    /** Evaluate polynomial {@code p} at {@code x} (Horner). */
    public static NDArray polyval(NDArray p, NDArray x) {
        NDArray pf = NPShape.ravel(p);
        NDArray out = new NDArray(DType.FLOAT64, x.shape);
        for (int i = 0; i < x.size; i++) {
            double xi = x.getDouble(i);
            double acc = 0;
            for (int k = 0; k < pf.size; k++) acc = acc * xi + pf.getDouble(k);
            out.setDouble(i, acc);
        }
        return out;
    }

    public static double polyval(NDArray p, double x) {
        return polyval(p, NP.array(new double[]{x})).getDouble(0);
    }

    /** Companion-matrix eigenvalue roots of monic polynomial. */
    public static NDArray roots(NDArray p) {
        NDArray pf = NPShape.ravel(p);
        // strip leading zeros
        int start = 0;
        while (start < pf.size && Math.abs(pf.getDouble(start)) < 1e-15) start++;
        if (start >= pf.size) return new NDArray(DType.COMPLEX128, 0);
        int n = (int) pf.size - start - 1; // degree
        if (n <= 0) return new NDArray(DType.COMPLEX128, 0);
        double lead = pf.getDouble(start);
        // companion matrix n x n
        NDArray C = NP.zeros(n, n);
        for (int i = 0; i < n - 1; i++) C.setDouble((i + 1) * n + i, 1.0);
        for (int j = 0; j < n; j++) {
            C.setDouble(j, -pf.getDouble(start + 1 + j) / lead);
        }
        // For general real companion use eig; return real parts as complex with 0 imag if real-only path
        NDArray[] ev = NPLinalg.eig(C);
        NDArray vals = ev[0];
        // promote to complex128
        double[] interleaved = new double[(int) vals.size * 2];
        for (int i = 0; i < vals.size; i++) {
            interleaved[i * 2] = vals.getDouble(i);
            interleaved[i * 2 + 1] = 0;
        }
        return new NDArray(interleaved, DType.COMPLEX128, vals.size);
    }

    /** Polynomial addition (highest-degree-first coefficients). */
    public static NDArray polyadd(NDArray p1, NDArray p2) {
        NDArray a = NPShape.ravel(p1);
        NDArray b = NPShape.ravel(p2);
        int n = (int) Math.max(a.size, b.size);
        NDArray out = NP.zeros(n);
        for (int i = 0; i < a.size; i++) out.setDouble((int) (n - a.size + i), a.getDouble(i));
        for (int i = 0; i < b.size; i++) {
            int idx = (int) (n - b.size + i);
            out.setDouble(idx, out.getDouble(idx) + b.getDouble(i));
        }
        return trimLeading(out);
    }

    public static NDArray polysub(NDArray p1, NDArray p2) {
        return polyadd(p1, NPMath.negative(NPShape.ravel(p2)));
    }

    public static NDArray polymul(NDArray p1, NDArray p2) {
        NDArray a = NPShape.ravel(p1);
        NDArray b = NPShape.ravel(p2);
        int n = (int) (a.size + b.size - 1);
        if (n <= 0) return NP.zeros(0);
        NDArray out = NP.zeros(n);
        for (int i = 0; i < a.size; i++) {
            for (int j = 0; j < b.size; j++) {
                int k = i + j;
                out.setDouble(k, out.getDouble(k) + a.getDouble(i) * b.getDouble(j));
            }
        }
        return trimLeading(out);
    }

    /** Formal derivative. */
    public static NDArray polyder(NDArray p, int m) {
        NDArray cur = NPShape.ravel(p);
        for (int d = 0; d < m; d++) {
            if (cur.size <= 1) return NP.zeros(1);
            int n = (int) cur.size - 1;
            NDArray out = new NDArray(DType.FLOAT64, n);
            for (int i = 0; i < n; i++) {
                out.setDouble(i, cur.getDouble(i) * (n - i));
            }
            cur = out;
        }
        return cur;
    }

    public static NDArray polyder(NDArray p) { return polyder(p, 1); }

    /** Formal anti-derivative; {@code k} integration constants (lowest terms first in k). */
    public static NDArray polyint(NDArray p, int m, double[] k) {
        NDArray cur = NPShape.ravel(p);
        for (int d = 0; d < m; d++) {
            int n = (int) cur.size;
            NDArray out = new NDArray(DType.FLOAT64, n + 1);
            for (int i = 0; i < n; i++) {
                double power = n - i; // degree of term after integration is power
                out.setDouble(i, cur.getDouble(i) / (n - i));
            }
            double c = (k != null && d < k.length) ? k[d] : 0;
            out.setDouble(n, c);
            cur = out;
        }
        return cur;
    }

    public static NDArray polyint(NDArray p) { return polyint(p, 1, new double[]{0}); }

    private static NDArray trimLeading(NDArray p) {
        int i = 0;
        while (i < p.size - 1 && Math.abs(p.getDouble(i)) < 1e-15) i++;
        if (i == 0) return p;
        double[] d = new double[(int) p.size - i];
        for (int j = 0; j < d.length; j++) d[j] = p.getDouble(i + j);
        return new NDArray(d);
    }

    /** Lightweight polynomial object (highest-degree-first coeffs). */
    public static final class Poly1d {
        public final NDArray c; // coefficients

        public Poly1d(NDArray coeffs) {
            this.c = trimLeading(NPShape.ravel(coeffs));
        }

        public Poly1d(double[] coeffs) {
            this(new NDArray(coeffs));
        }

        public int order() { return Math.max(0, (int) c.size - 1); }

        public double call(double x) { return polyval(c, x); }

        public NDArray call(NDArray x) { return polyval(c, x); }

        public Poly1d deriv() { return new Poly1d(polyder(c)); }

        public Poly1d integ() { return new Poly1d(polyint(c)); }

        public NDArray roots() { return NPPoly.roots(c); }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("poly1d([");
            for (int i = 0; i < c.size; i++) {
                if (i > 0) sb.append(", ");
                sb.append(c.getDouble(i));
            }
            sb.append("])");
            return sb.toString();
        }
    }

    public static Poly1d poly1d(NDArray c) { return new Poly1d(c); }
    public static Poly1d poly1d(double[] c) { return new Poly1d(c); }
}
