package org.bytedeco.pytorch.data.numpy;

/**
 * NumPy-style linear algebra: matmul, decompositions, norms, solvers.
 * Pure-Java implementations suitable for moderate sizes (education / binding layer).
 */
public final class NPLinalg {
    private NPLinalg() {}

    public static NDArray dot(NDArray a, NDArray b) {
        if (a.shape.length == 1 && b.shape.length == 1) {
            if (a.size != b.size) throw new IllegalArgumentException("dot size mismatch");
            double s = 0;
            for (int i = 0; i < a.size; i++) s += a.getDouble(i) * b.getDouble(i);
            NDArray out = new NDArray(DType.FLOAT64);
            out.setDouble(0, s);
            return out;
        }
        if (a.shape.length == 1 && b.shape.length == 2) {
            return NPShape.squeeze(matmul(NPShape.reshape(a, 1, a.size), b), 0);
        }
        if (a.shape.length == 2 && b.shape.length == 1) {
            return NPShape.squeeze(matmul(a, NPShape.reshape(b, b.size, 1)), 1);
        }
        return matmul(a, b);
    }

    public static NDArray matmul(NDArray a, NDArray b) {
        if (a.shape.length == 1 && b.shape.length == 1) return dot(a, b);
        NDArray aa = a.shape.length == 1 ? NPShape.reshape(a, 1, a.size) : a;
        NDArray bb = b.shape.length == 1 ? NPShape.reshape(b, b.size, 1) : b;
        if (aa.shape.length != 2 || bb.shape.length != 2) {
            // batch matmul: treat leading dims as batch if last-2 match
            return batchMatmul(aa, bb);
        }
        long m = aa.shape[0], k = aa.shape[1], k2 = bb.shape[0], n = bb.shape[1];
        if (k != k2) throw new IllegalArgumentException("matmul inner dims mismatch: " + k + " vs " + k2);
        NDArray out = new NDArray(DType.FLOAT64, m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int t = 0; t < k; t++) {
                    s += aa.getDouble((int) (i * k + t)) * bb.getDouble((int) (t * n + j));
                }
                out.setDouble((int) (i * n + j), s);
            }
        }
        if (a.shape.length == 1) return NPShape.squeeze(out, 0);
        if (b.shape.length == 1) return NPShape.squeeze(out, 1);
        return out;
    }

    private static NDArray batchMatmul(NDArray a, NDArray b) {
        // Require a[..., m, k] @ b[..., k, n]
        if (a.shape.length < 2 || b.shape.length < 2) {
            throw new IllegalArgumentException("matmul requires at least 1D");
        }
        long m = a.shape[a.shape.length - 2];
        long k = a.shape[a.shape.length - 1];
        long k2 = b.shape[b.shape.length - 2];
        long n = b.shape[b.shape.length - 1];
        if (k != k2) throw new IllegalArgumentException("matmul inner dims mismatch");
        long[] aBatch = ArraysCopy(a.shape, 0, a.shape.length - 2);
        long[] bBatch = ArraysCopy(b.shape, 0, b.shape.length - 2);
        long[] batch = NPArrayUtil.broadcastShapes(aBatch, bBatch);
        long[] outShape = new long[batch.length + 2];
        System.arraycopy(batch, 0, outShape, 0, batch.length);
        outShape[batch.length] = m;
        outShape[batch.length + 1] = n;
        NDArray out = new NDArray(DType.FLOAT64, outShape);
        long batchN = NPArrayUtil.numel(batch.length == 0 ? new long[]{1} : batch);
        if (batch.length == 0) batchN = 1;
        long aMat = m * k, bMat = k * n, oMat = m * n;
        // For each batch index, extract matrices — simplified flat when no batch
        if (batch.length == 0) {
            return matmul(NPShape.reshape(a, m, k), NPShape.reshape(b, k, n));
        }
        long[] bSt = NPArrayUtil.stridesOf(batch);
        long[] aBSt = aBatch.length == 0 ? new long[0] : NPArrayUtil.stridesOf(aBatch);
        long[] bBSt = bBatch.length == 0 ? new long[0] : NPArrayUtil.stridesOf(bBatch);
        int[] bIdx = new int[batch.length];
        for (int bi = 0; bi < batchN; bi++) {
            if (batch.length > 0) NPArrayUtil.fillMultiIndex(bi, batch, bSt, bIdx);
            int aOff = aBatch.length == 0 ? 0 : NPArrayUtil.broadcastIndex(bIdx, aBatch, aBSt);
            int bOff = bBatch.length == 0 ? 0 : NPArrayUtil.broadcastIndex(bIdx, bBatch, bBSt);
            // aOff/bOff are indices into batch grid; convert to element offset
            aOff = (int) (aOff * aMat);
            bOff = (int) (bOff * bMat);
            int oOff = (int) (bi * oMat);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double s = 0;
                    for (int t = 0; t < k; t++) {
                        s += a.getDouble(aOff + (int) (i * k + t)) * b.getDouble(bOff + (int) (t * n + j));
                    }
                    out.setDouble(oOff + (int) (i * n + j), s);
                }
            }
        }
        return out;
    }

    private static long[] ArraysCopy(long[] src, int from, int to) {
        long[] o = new long[to - from];
        System.arraycopy(src, from, o, 0, o.length);
        return o;
    }

    public static NDArray tensordot(NDArray a, NDArray b, int axes) {
        // Contract last `axes` dims of a with first `axes` dims of b
        if (axes < 0) throw new IllegalArgumentException("axes");
        if (a.shape.length < axes || b.shape.length < axes) {
            throw new IllegalArgumentException("tensordot axes too large");
        }
        long contract = 1;
        for (int i = 0; i < axes; i++) {
            long da = a.shape[a.shape.length - axes + i];
            long db = b.shape[i];
            if (da != db) throw new IllegalArgumentException("tensordot dim mismatch");
            contract *= da;
        }
        long aFront = a.size / contract;
        long bBack = b.size / contract;
        NDArray flatA = NPShape.reshape(a, aFront, contract);
        NDArray flatB = NPShape.reshape(b, contract, bBack);
        NDArray prod = matmul(flatA, flatB);
        long[] outShape = new long[a.shape.length - axes + b.shape.length - axes];
        int p = 0;
        for (int i = 0; i < a.shape.length - axes; i++) outShape[p++] = a.shape[i];
        for (int i = axes; i < b.shape.length; i++) outShape[p++] = b.shape[i];
        if (outShape.length == 0) return prod; // scalar
        return NPShape.reshape(prod, outShape);
    }

    public static double trace(NDArray a) {
        if (a.shape.length != 2) throw new IllegalArgumentException("trace expects 2D");
        int n = (int) Math.min(a.shape[0], a.shape[1]);
        double s = 0;
        for (int i = 0; i < n; i++) s += a.getDouble(i * (int) a.shape[1] + i);
        return s;
    }

    public static NDArray vander(NDArray x, Integer N, boolean increasing) {
        int n = N == null ? (int) x.size : N;
        NDArray out = new NDArray(DType.FLOAT64, x.size, n);
        for (int i = 0; i < x.size; i++) {
            double xi = x.getDouble(i);
            double p = 1;
            if (increasing) {
                for (int j = 0; j < n; j++) {
                    out.setDouble((int) (i * n + j), p);
                    p *= xi;
                }
            } else {
                for (int j = n - 1; j >= 0; j--) {
                    out.setDouble((int) (i * n + j), p);
                    p *= xi;
                }
            }
        }
        return out;
    }

    public static NDArray vander(NDArray x) { return vander(x, null, false); }

    public static NDArray matrix_power(NDArray a, int n) {
        requireSquare(a);
        if (n == 0) return NP.eye((int) a.shape[0]);
        if (n < 0) return matrix_power(inv(a), -n);
        NDArray result = NP.eye((int) a.shape[0]);
        NDArray base = a;
        int e = n;
        while (e > 0) {
            if ((e & 1) == 1) result = matmul(result, base);
            base = matmul(base, base);
            e >>= 1;
        }
        return result;
    }

    public static NDArray inv(NDArray a) {
        requireSquare(a);
        int n = (int) a.shape[0];
        // Gauss-Jordan on [A|I]
        double[] m = new double[n * 2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) m[i * 2 * n + j] = a.getDouble(i * n + j);
            m[i * 2 * n + n + i] = 1.0;
        }
        for (int col = 0; col < n; col++) {
            int piv = col;
            double best = Math.abs(m[piv * 2 * n + col]);
            for (int r = col + 1; r < n; r++) {
                double v = Math.abs(m[r * 2 * n + col]);
                if (v > best) { best = v; piv = r; }
            }
            if (best < 1e-15) throw new ArithmeticException("singular matrix");
            if (piv != col) swapRows(m, col, piv, 2 * n);
            double div = m[col * 2 * n + col];
            for (int j = 0; j < 2 * n; j++) m[col * 2 * n + j] /= div;
            for (int r = 0; r < n; r++) {
                if (r == col) continue;
                double f = m[r * 2 * n + col];
                for (int j = 0; j < 2 * n; j++) m[r * 2 * n + j] -= f * m[col * 2 * n + j];
            }
        }
        NDArray out = new NDArray(DType.FLOAT64, n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                out.setDouble(i * n + j, m[i * 2 * n + n + j]);
        return out;
    }

    public static NDArray pinv(NDArray a, double rcond) {
        // Via SVD: V S+ U^T
        NDArray[] usv = svd(a, true, true);
        NDArray U = usv[0], S = usv[1], Vt = usv[2];
        double smax = 0;
        for (int i = 0; i < S.size; i++) smax = Math.max(smax, S.getDouble(i));
        double tol = rcond * smax;
        NDArray Sp = new NDArray(DType.FLOAT64, S.shape);
        for (int i = 0; i < S.size; i++) {
            double s = S.getDouble(i);
            Sp.setDouble(i, s > tol ? 1.0 / s : 0.0);
        }
        // pinv = V @ diag(Sp) @ U.T  ; Vt is V^T so V = Vt.T
        NDArray V = NPShape.transpose(Vt);
        int k = (int) Sp.size;
        NDArray mid = new NDArray(DType.FLOAT64, V.shape[0], U.shape[0]);
        // V[:, :k] * Sp * U[:, :k].T
        int vRows = (int) V.shape[0];
        int uRows = (int) U.shape[0];
        for (int i = 0; i < vRows; i++) {
            for (int j = 0; j < uRows; j++) {
                double s = 0;
                for (int t = 0; t < k; t++) {
                    s += V.getDouble(i * k + t) * Sp.getDouble(t) * U.getDouble(j * (int) U.shape[1] + t);
                }
                mid.setDouble(i * uRows + j, s);
            }
        }
        // Careful with shapes when economy SVD
        return mid;
    }

    public static NDArray pinv(NDArray a) { return pinv(a, 1e-15); }

    public static double det(NDArray a) {
        requireSquare(a);
        int n = (int) a.shape[0];
        double[] m = new double[n * n];
        for (int i = 0; i < n * n; i++) m[i] = a.getDouble(i);
        double det = 1;
        for (int col = 0; col < n; col++) {
            int piv = col;
            for (int r = col + 1; r < n; r++)
                if (Math.abs(m[r * n + col]) > Math.abs(m[piv * n + col])) piv = r;
            if (Math.abs(m[piv * n + col]) < 1e-15) return 0;
            if (piv != col) {
                swapRows(m, col, piv, n);
                det = -det;
            }
            det *= m[col * n + col];
            for (int r = col + 1; r < n; r++) {
                double f = m[r * n + col] / m[col * n + col];
                for (int j = col; j < n; j++) m[r * n + j] -= f * m[col * n + j];
            }
        }
        return det;
    }

    public static NDArray solve(NDArray a, NDArray b) {
        requireSquare(a);
        int n = (int) a.shape[0];
        NDArray bb = b.shape.length == 1 ? NPShape.reshape(b, n, 1) : b;
        if (bb.shape[0] != n) throw new IllegalArgumentException("b rows mismatch");
        int nrhs = (int) bb.shape[1];
        double[] m = new double[n * (n + nrhs)];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) m[i * (n + nrhs) + j] = a.getDouble(i * n + j);
            for (int j = 0; j < nrhs; j++) m[i * (n + nrhs) + n + j] = bb.getDouble(i * nrhs + j);
        }
        int w = n + nrhs;
        for (int col = 0; col < n; col++) {
            int piv = col;
            for (int r = col + 1; r < n; r++)
                if (Math.abs(m[r * w + col]) > Math.abs(m[piv * w + col])) piv = r;
            if (Math.abs(m[piv * w + col]) < 1e-15) throw new ArithmeticException("singular");
            if (piv != col) swapRows(m, col, piv, w);
            double div = m[col * w + col];
            for (int j = col; j < w; j++) m[col * w + j] /= div;
            for (int r = 0; r < n; r++) {
                if (r == col) continue;
                double f = m[r * w + col];
                for (int j = col; j < w; j++) m[r * w + j] -= f * m[col * w + j];
            }
        }
        NDArray out = new NDArray(DType.FLOAT64, b.shape.length == 1 ? new long[]{n} : new long[]{n, nrhs});
        for (int i = 0; i < n; i++)
            for (int j = 0; j < nrhs; j++)
                out.setDouble(i * nrhs + j, m[i * w + n + j]);
        return out;
    }

    public static NDArray[] lstsq(NDArray a, NDArray b) {
        // Normal equations: (A^T A) x = A^T b  (simple; not rank-revealing)
        NDArray At = NPShape.transpose(a);
        NDArray AtA = matmul(At, a);
        NDArray bb = b.shape.length == 1 ? NPShape.reshape(b, b.size, 1) : b;
        NDArray Atb = matmul(At, bb);
        NDArray x = solve(AtA, Atb);
        NDArray resid = NPMath.subtract(matmul(a, x.shape.length == 1 ? NPShape.reshape(x, x.size, 1) : x), bb);
        double r = 0;
        for (int i = 0; i < resid.size; i++) r += resid.getDouble(i) * resid.getDouble(i);
        NDArray residuals = NP.array(new double[]{r});
        NDArray rank = NP.array(new long[]{a.shape[1]}, DType.INT64);
        NDArray singular = NP.zeros((int) Math.min(a.shape[0], a.shape[1]));
        return new NDArray[]{x, residuals, rank, singular};
    }

    public static NDArray cholesky(NDArray a) {
        requireSquare(a);
        int n = (int) a.shape[0];
        NDArray L = new NDArray(DType.FLOAT64, n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double s = 0;
                for (int k = 0; k < j; k++) s += L.getDouble(i * n + k) * L.getDouble(j * n + k);
                if (i == j) {
                    double v = a.getDouble(i * n + i) - s;
                    if (v <= 0) throw new ArithmeticException("matrix not SPD");
                    L.setDouble(i * n + j, Math.sqrt(v));
                } else {
                    L.setDouble(i * n + j, (a.getDouble(i * n + j) - s) / L.getDouble(j * n + j));
                }
            }
        }
        return L;
    }

    public static NDArray[] qr(NDArray a) {
        // Classical Gram-Schmidt
        int m = (int) a.shape[0], n = (int) a.shape[1];
        NDArray Q = new NDArray(DType.FLOAT64, m, n);
        NDArray R = new NDArray(DType.FLOAT64, n, n);
        double[] v = new double[m];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) v[i] = a.getDouble(i * n + j);
            for (int k = 0; k < j; k++) {
                double dot = 0;
                for (int i = 0; i < m; i++) dot += Q.getDouble(i * n + k) * a.getDouble(i * n + j);
                R.setDouble(k * n + j, dot);
                for (int i = 0; i < m; i++) v[i] -= dot * Q.getDouble(i * n + k);
            }
            double norm = 0;
            for (int i = 0; i < m; i++) norm += v[i] * v[i];
            norm = Math.sqrt(norm);
            R.setDouble(j * n + j, norm);
            if (norm < 1e-15) norm = 1;
            for (int i = 0; i < m; i++) Q.setDouble(i * n + j, v[i] / norm);
        }
        return new NDArray[]{Q, R};
    }

    /**
     * Compact SVD via eigenvalue decomposition of A^T A (or A A^T).
     * Returns U, S, Vt.
     */
    public static NDArray[] svd(NDArray a, boolean fullMatrices, boolean computeUv) {
        int m = (int) a.shape[0], n = (int) a.shape[1];
        boolean thinA = m >= n;
        NDArray ata = thinA ? matmul(NPShape.transpose(a), a) : matmul(a, NPShape.transpose(a));
        NDArray[] eig = eigh(ata); // vals ascending, vecs as columns
        NDArray vals = eig[0];
        NDArray vecs = eig[1];
        int k = (int) Math.min(m, n);
        // reverse to descending
        double[] s = new double[k];
        for (int i = 0; i < k; i++) {
            double v = vals.getDouble((int) vals.size - 1 - i);
            s[i] = Math.sqrt(Math.max(v, 0));
        }
        NDArray S = new NDArray(s);
        if (!computeUv) return new NDArray[]{null, S, null};

        int dim = (int) ata.shape[0];
        NDArray Vrev = new NDArray(DType.FLOAT64, dim, k);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < k; j++) {
                Vrev.setDouble(i * k + j, vecs.getDouble(i * dim + (dim - 1 - j)));
            }
        }
        NDArray U, Vt;
        if (thinA) {
            // V from A^T A; U = A V S^{-1}
            Vt = NPShape.transpose(Vrev);
            U = new NDArray(DType.FLOAT64, m, k);
            NDArray AV = matmul(a, Vrev);
            for (int j = 0; j < k; j++) {
                double sj = s[j] < 1e-15 ? 1 : s[j];
                for (int i = 0; i < m; i++) U.setDouble(i * k + j, AV.getDouble(i * k + j) / sj);
            }
        } else {
            U = Vrev;
            NDArray AtU = matmul(NPShape.transpose(a), U);
            NDArray V = new NDArray(DType.FLOAT64, n, k);
            for (int j = 0; j < k; j++) {
                double sj = s[j] < 1e-15 ? 1 : s[j];
                for (int i = 0; i < n; i++) V.setDouble(i * k + j, AtU.getDouble(i * k + j) / sj);
            }
            Vt = NPShape.transpose(V);
        }
        return new NDArray[]{U, S, Vt};
    }

    public static NDArray[] svd(NDArray a) { return svd(a, false, true); }

    /** Symmetric eigendecomposition via Jacobi rotations. Returns (eigenvalues asc, eigenvectors as columns). */
    public static NDArray[] eigh(NDArray a) {
        requireSquare(a);
        int n = (int) a.shape[0];
        double[] A = new double[n * n];
        for (int i = 0; i < n * n; i++) A[i] = a.getDouble(i);
        double[] V = new double[n * n];
        for (int i = 0; i < n; i++) V[i * n + i] = 1;
        for (int iter = 0; iter < 100 * n * n; iter++) {
            // find max off-diag
            int p = 0, q = 1;
            double max = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double v = Math.abs(A[i * n + j]);
                    if (v > max) { max = v; p = i; q = j; }
                }
            }
            if (max < 1e-12) break;
            double app = A[p * n + p], aqq = A[q * n + q], apq = A[p * n + q];
            double tau = (aqq - app) / (2 * apq);
            double t = Math.signum(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
            if (tau == 0) t = 1;
            double c = 1 / Math.sqrt(1 + t * t);
            double s = t * c;
            for (int i = 0; i < n; i++) {
                if (i == p || i == q) continue;
                double aip = A[i * n + p], aiq = A[i * n + q];
                A[i * n + p] = A[p * n + i] = c * aip - s * aiq;
                A[i * n + q] = A[q * n + i] = c * aiq + s * aip;
            }
            A[p * n + p] = app - t * apq;
            A[q * n + q] = aqq + t * apq;
            A[p * n + q] = A[q * n + p] = 0;
            for (int i = 0; i < n; i++) {
                double vip = V[i * n + p], viq = V[i * n + q];
                V[i * n + p] = c * vip - s * viq;
                V[i * n + q] = c * viq + s * vip;
            }
        }
        // extract + sort ascending
        Integer[] order = new Integer[n];
        double[] vals = new double[n];
        for (int i = 0; i < n; i++) { vals[i] = A[i * n + i]; order[i] = i; }
        ArraysSort(order, vals);
        NDArray w = new NDArray(DType.FLOAT64, n);
        NDArray vecs = new NDArray(DType.FLOAT64, n, n);
        for (int j = 0; j < n; j++) {
            w.setDouble(j, vals[order[j]]);
            for (int i = 0; i < n; i++) vecs.setDouble(i * n + j, V[i * n + order[j]]);
        }
        return new NDArray[]{w, vecs};
    }

    public static NDArray[] eig(NDArray a) {
        // For general matrices, fall back to a simple QR algorithm (real parts only emphasis)
        requireSquare(a);
        // If roughly symmetric, use eigh
        boolean sym = true;
        int n = (int) a.shape[0];
        for (int i = 0; i < n && sym; i++)
            for (int j = i + 1; j < n; j++)
                if (Math.abs(a.getDouble(i * n + j) - a.getDouble(j * n + i)) > 1e-9) sym = false;
        if (sym) return eigh(a);
        // Unsymmetric: return diagonal of Schur via basic QR iters (eigenvalues only reliable)
        NDArray Ak = NPShape.copy(a);
        NDArray Qacc = NP.eye(n);
        for (int it = 0; it < 50 * n; it++) {
            NDArray[] qr = qr(Ak);
            Ak = matmul(qr[1], qr[0]);
            Qacc = matmul(Qacc, qr[0]);
        }
        NDArray vals = new NDArray(DType.FLOAT64, n);
        for (int i = 0; i < n; i++) vals.setDouble(i, Ak.getDouble(i * n + i));
        return new NDArray[]{vals, Qacc};
    }

    public static NDArray norm(NDArray x, String ord, Integer axis, boolean keepdims) {
        if (axis == null && x.shape.length <= 1) {
            double s = 0;
            if (ord == null || "2".equals(ord) || "fro".equals(ord)) {
                for (int i = 0; i < x.size; i++) s += x.getDouble(i) * x.getDouble(i);
                s = Math.sqrt(s);
            } else if ("1".equals(ord)) {
                for (int i = 0; i < x.size; i++) s += Math.abs(x.getDouble(i));
            } else if ("inf".equals(ord)) {
                s = 0;
                for (int i = 0; i < x.size; i++) s = Math.max(s, Math.abs(x.getDouble(i)));
            } else if ("-inf".equals(ord)) {
                s = Double.POSITIVE_INFINITY;
                for (int i = 0; i < x.size; i++) s = Math.min(s, Math.abs(x.getDouble(i)));
            } else if ("0".equals(ord)) {
                for (int i = 0; i < x.size; i++) if (x.getDouble(i) != 0) s++;
            } else {
                double p = Double.parseDouble(ord);
                for (int i = 0; i < x.size; i++) s += Math.pow(Math.abs(x.getDouble(i)), p);
                s = Math.pow(s, 1.0 / p);
            }
            NDArray out = new NDArray(DType.FLOAT64);
            out.setDouble(0, s);
            return out;
        }
        if (axis == null && x.shape.length == 2 && (ord == null || "fro".equals(ord) || "2".equals(ord))) {
            // Frobenius
            double s = 0;
            for (int i = 0; i < x.size; i++) s += x.getDouble(i) * x.getDouble(i);
            NDArray out = new NDArray(DType.FLOAT64);
            out.setDouble(0, Math.sqrt(s));
            return out;
        }
        // axis vector norm L2
        NDArray sq = NPMath.square(x);
        NDArray summed = NPReduce.sum(sq, axis, keepdims);
        return NPMath.sqrt(summed);
    }

    public static NDArray norm(NDArray x) { return norm(x, null, null, false); }

    public static NDArray norm(NDArray x, Integer axis) { return norm(x, "2", axis, false); }

    private static void requireSquare(NDArray a) {
        if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
            throw new IllegalArgumentException("expected square 2D matrix");
        }
    }

    private static void swapRows(double[] m, int r1, int r2, int cols) {
        for (int j = 0; j < cols; j++) {
            double t = m[r1 * cols + j];
            m[r1 * cols + j] = m[r2 * cols + j];
            m[r2 * cols + j] = t;
        }
    }

    private static void ArraysSort(Integer[] order, double[] vals) {
        java.util.Arrays.sort(order, (i, j) -> Double.compare(vals[i], vals[j]));
    }
}
