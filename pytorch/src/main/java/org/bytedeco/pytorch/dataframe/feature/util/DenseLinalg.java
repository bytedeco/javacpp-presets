package org.bytedeco.pytorch.dataframe.feature.util;

import java.util.Arrays;
import java.util.Random;

/**
 * Dense linear algebra helpers for feature transformers (PCA / LDA / FA / NMF).
 * Pure Java — no native BLAS required.
 */
public final class DenseLinalg {
    private DenseLinalg() {}

    public static final class EigenResult {
        /** Eigenvalues sorted descending. */
        public final double[] eigenvalues;
        /** Eigenvectors as rows: vectors[i] corresponds to eigenvalues[i], unit length. */
        public final double[][] vectors;

        public EigenResult(double[] eigenvalues, double[][] vectors) {
            this.eigenvalues = eigenvalues;
            this.vectors = vectors;
        }
    }

    /** Jacobi eigenvalue decomposition for symmetric matrix. Returns top eigenvalues desc. */
    public static EigenResult eighSymmetric(double[][] A) {
        int n = A.length;
        double[][] a = copy(A);
        // Ensure symmetry
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double s = 0.5 * (a[i][j] + a[j][i]);
                a[i][j] = a[j][i] = s;
            }
        }
        double[][] v = identity(n);
        final int maxSweeps = 64;
        final double tol = 1e-12;

        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            double off = 0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    off += a[i][j] * a[i][j];
            if (Math.sqrt(off) < tol * n) break;

            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    double apq = a[p][q];
                    if (Math.abs(apq) < 1e-15) continue;
                    double app = a[p][p], aqq = a[q][q];
                    double theta = 0.5 * (aqq - app) / apq;
                    double t;
                    if (Math.abs(theta) > 1e12) {
                        t = 0.5 / theta;
                    } else {
                        t = Math.copySign(1.0, theta) / (Math.abs(theta) + Math.sqrt(1.0 + theta * theta));
                        if (theta == 0.0) t = 1.0;
                    }
                    double c = 1.0 / Math.sqrt(1.0 + t * t);
                    double s = t * c;

                    // rotate A
                    a[p][p] = app - t * apq;
                    a[q][q] = aqq + t * apq;
                    a[p][q] = a[q][p] = 0.0;
                    for (int r = 0; r < n; r++) {
                        if (r == p || r == q) continue;
                        double arp = a[r][p], arq = a[r][q];
                        a[r][p] = a[p][r] = c * arp - s * arq;
                        a[r][q] = a[q][r] = c * arq + s * arp;
                    }
                    // rotate V
                    for (int r = 0; r < n; r++) {
                        double vrp = v[r][p], vrq = v[r][q];
                        v[r][p] = c * vrp - s * vrq;
                        v[r][q] = c * vrq + s * vrp;
                    }
                }
            }
        }

        double[] evals = new double[n];
        for (int i = 0; i < n; i++) evals[i] = a[i][i];

        // sort descending by eigenvalue; eigenvectors currently as columns of v
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (i, j) -> Double.compare(evals[j], evals[i]));

        double[] sortedE = new double[n];
        double[][] sortedV = new double[n][n]; // rows = eigenvectors
        for (int i = 0; i < n; i++) {
            int idx = order[i];
            sortedE[i] = evals[idx];
            double norm = 0;
            for (int r = 0; r < n; r++) {
                sortedV[i][r] = v[r][idx];
                norm += sortedV[i][r] * sortedV[i][r];
            }
            norm = Math.sqrt(norm);
            if (norm > 1e-15) {
                for (int r = 0; r < n; r++) sortedV[i][r] /= norm;
            }
            // deterministic sign: first nonzero entry >= 0
            for (int r = 0; r < n; r++) {
                if (Math.abs(sortedV[i][r]) > 1e-12) {
                    if (sortedV[i][r] < 0) {
                        for (int c = 0; c < n; c++) sortedV[i][c] = -sortedV[i][c];
                    }
                    break;
                }
            }
        }
        return new EigenResult(sortedE, sortedV);
    }

    public static double[][] identity(int n) {
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++) I[i][i] = 1.0;
        return I;
    }

    public static double[][] copy(double[][] A) {
        double[][] B = new double[A.length][];
        for (int i = 0; i < A.length; i++) B[i] = A[i].clone();
        return B;
    }

    public static double[][] transpose(double[][] A) {
        int n = A.length, m = A[0].length;
        double[][] T = new double[m][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                T[j][i] = A[i][j];
        return T;
    }

    public static double[][] matmul(double[][] A, double[][] B) {
        int n = A.length, p = A[0].length, m = B[0].length;
        if (B.length != p) throw new IllegalArgumentException("matmul shape mismatch");
        double[][] C = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < p; k++) {
                double aik = A[i][k];
                if (aik == 0) continue;
                for (int j = 0; j < m; j++) C[i][j] += aik * B[k][j];
            }
        }
        return C;
    }

    /** A @ B^T */
    public static double[][] matmulBT(double[][] A, double[][] B) {
        int n = A.length, p = A[0].length, m = B.length;
        if (B[0].length != p) throw new IllegalArgumentException("matmulBT shape mismatch");
        double[][] C = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double s = 0;
                for (int k = 0; k < p; k++) s += A[i][k] * B[j][k];
                C[i][j] = s;
            }
        }
        return C;
    }

    /** A^T @ B */
    public static double[][] matmulAT(double[][] A, double[][] B) {
        int p = A.length, n = A[0].length, m = B[0].length;
        if (B.length != p) throw new IllegalArgumentException("matmulAT shape mismatch");
        double[][] C = new double[n][m];
        for (int k = 0; k < p; k++) {
            for (int i = 0; i < n; i++) {
                double aki = A[k][i];
                if (aki == 0) continue;
                for (int j = 0; j < m; j++) C[i][j] += aki * B[k][j];
            }
        }
        return C;
    }

    public static double[] matvec(double[][] A, double[] x) {
        int n = A.length, m = A[0].length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < m; j++) s += A[i][j] * x[j];
            y[i] = s;
        }
        return y;
    }

    public static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    public static double norm2(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    public static void scaleInPlace(double[] a, double s) {
        for (int i = 0; i < a.length; i++) a[i] *= s;
    }

    public static double[][] covariance(double[][] Xcentered, boolean sample) {
        int n = Xcentered.length, d = Xcentered[0].length;
        double[][] cov = new double[d][d];
        double denom = sample ? Math.max(1, n - 1) : Math.max(1, n);
        for (int i = 0; i < n; i++) {
            double[] row = Xcentered[i];
            for (int j = 0; j < d; j++) {
                double rj = row[j];
                for (int k = j; k < d; k++) {
                    cov[j][k] += rj * row[k];
                }
            }
        }
        for (int j = 0; j < d; j++) {
            for (int k = j; k < d; k++) {
                cov[j][k] /= denom;
                cov[k][j] = cov[j][k];
            }
        }
        return cov;
    }

    public static double[] mean(double[][] X) {
        int n = X.length, d = X[0].length;
        double[] m = new double[d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) m[j] += X[i][j];
        for (int j = 0; j < d; j++) m[j] /= n;
        return m;
    }

    public static double[][] center(double[][] X, double[] mean) {
        int n = X.length, d = X[0].length;
        double[][] C = new double[n][d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) C[i][j] = X[i][j] - mean[j];
        return C;
    }

    /** Solve symmetric positive-definite system via Gauss with partial pivoting (general). */
    public static double[] solve(double[][] A, double[] b) {
        int n = A.length;
        double[][] M = copy(A);
        double[] x = b.clone();
        for (int k = 0; k < n; k++) {
            int piv = k;
            double best = Math.abs(M[k][k]);
            for (int i = k + 1; i < n; i++) {
                double v = Math.abs(M[i][k]);
                if (v > best) { best = v; piv = i; }
            }
            if (best < 1e-15) {
                // ridge fallback
                M[k][k] += 1e-8;
            }
            if (piv != k) {
                double[] tmp = M[k]; M[k] = M[piv]; M[piv] = tmp;
                double t = x[k]; x[k] = x[piv]; x[piv] = t;
            }
            double diag = M[k][k];
            for (int i = k + 1; i < n; i++) {
                double f = M[i][k] / diag;
                x[i] -= f * x[k];
                for (int j = k; j < n; j++) M[i][j] -= f * M[k][j];
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            double s = x[i];
            for (int j = i + 1; j < n; j++) s -= M[i][j] * x[j];
            x[i] = s / (Math.abs(M[i][i]) < 1e-15 ? 1e-15 : M[i][i]);
        }
        return x;
    }

    /** Solve A X = B column-wise. */
    public static double[][] solveMulti(double[][] A, double[][] B) {
        int m = B[0].length;
        double[][] X = new double[A.length][m];
        for (int j = 0; j < m; j++) {
            double[] b = new double[B.length];
            for (int i = 0; i < B.length; i++) b[i] = B[i][j];
            double[] x = solve(A, b);
            for (int i = 0; i < x.length; i++) X[i][j] = x[i];
        }
        return X;
    }

    public static double[][] randomPositive(int rows, int cols, long seed) {
        Random rng = new Random(seed);
        double[][] M = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                M[i][j] = Math.abs(rng.nextGaussian()) + 1e-4;
        return M;
    }

    /** Element-wise max with eps (for NMF stability). */
    public static void clipMinInPlace(double[][] M, double eps) {
        for (double[] row : M)
            for (int j = 0; j < row.length; j++)
                if (row[j] < eps) row[j] = eps;
    }
}
