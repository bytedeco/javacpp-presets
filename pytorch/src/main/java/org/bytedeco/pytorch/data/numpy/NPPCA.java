package org.bytedeco.pytorch.data.numpy;

/**
 * Principal Component Analysis on 2D data (n_samples × n_features).
 */
public final class NPPCA {
    private NPPCA() {}

    public static final class Result {
        /** Centered-data projection: (n_samples × n_components). */
        public final NDArray transformed;
        /** Principal axes (eigenvectors) as rows: (n_components × n_features). */
        public final NDArray components;
        /** Explained variance per component. */
        public final NDArray explainedVariance;
        /** Explained variance ratio. */
        public final NDArray explainedVarianceRatio;
        /** Feature-wise mean used for centering. */
        public final NDArray mean;
        /** Singular values (sqrt of eigenvalues * (n-1)). */
        public final NDArray singularValues;

        Result(NDArray transformed, NDArray components, NDArray explainedVariance,
               NDArray explainedVarianceRatio, NDArray mean, NDArray singularValues) {
            this.transformed = transformed;
            this.components = components;
            this.explainedVariance = explainedVariance;
            this.explainedVarianceRatio = explainedVarianceRatio;
            this.mean = mean;
            this.singularValues = singularValues;
        }
    }

    /**
     * Fit PCA on {@code X} shaped (n_samples, n_features) and transform.
     *
     * @param nComponents number of components; null → min(n_samples, n_features)
     */
    public static Result fitTransform(NDArray X, Integer nComponents) {
        if (X.shape.length != 2) throw new IllegalArgumentException("PCA expects 2D array");
        int n = (int) X.shape[0];
        int d = (int) X.shape[1];
        int k = nComponents == null ? Math.min(n, d) : nComponents;
        if (k <= 0 || k > Math.min(n, d)) {
            throw new IllegalArgumentException("n_components out of range");
        }

        NDArray mean = NPReduce.mean(X, 0, false); // (d,)
        NDArray centered = NPMath.subtract(X, NPShape.broadcast_to(NPShape.reshape(mean, 1, d), n, d));

        // covariance (d x d) with ddof=1
        NDArray cov;
        if (n >= d) {
            NDArray ct = NPShape.transpose(centered);
            cov = NPMath.multiply(NPLinalg.matmul(ct, centered), 1.0 / Math.max(n - 1, 1));
        } else {
            // use Gram matrix path via SVD of centered
            cov = null;
        }

        NDArray components;
        NDArray explainedVar;
        NDArray singular;

        if (cov != null) {
            NDArray[] eig = NPLinalg.eigh(cov); // ascending
            NDArray vals = eig[0];
            NDArray vecs = eig[1]; // columns
            // take top-k descending
            components = new NDArray(DType.FLOAT64, k, d);
            explainedVar = new NDArray(DType.FLOAT64, k);
            singular = new NDArray(DType.FLOAT64, k);
            for (int i = 0; i < k; i++) {
                int src = (int) vals.size - 1 - i;
                double ev = Math.max(vals.getDouble(src), 0);
                explainedVar.setDouble(i, ev);
                singular.setDouble(i, Math.sqrt(ev * Math.max(n - 1, 1)));
                for (int j = 0; j < d; j++) {
                    components.setDouble(i * d + j, vecs.getDouble(j * (int) vecs.shape[1] + src));
                }
            }
        } else {
            // SVD on centered (n x d): U S Vt
            NDArray[] usv = NPLinalg.svd(centered, false, true);
            NDArray S = usv[1];
            NDArray Vt = usv[2];
            components = new NDArray(DType.FLOAT64, k, d);
            explainedVar = new NDArray(DType.FLOAT64, k);
            singular = new NDArray(DType.FLOAT64, k);
            int vCols = (int) Vt.shape[1];
            for (int i = 0; i < k; i++) {
                double s = i < S.size ? S.getDouble(i) : 0;
                singular.setDouble(i, s);
                explainedVar.setDouble(i, (s * s) / Math.max(n - 1, 1));
                for (int j = 0; j < d; j++) {
                    // Vt is (k_full x d) roughly
                    components.setDouble(i * d + j, Vt.getDouble(i * vCols + j));
                }
            }
        }

        double total = 0;
        for (int i = 0; i < explainedVar.size; i++) total += explainedVar.getDouble(i);
        // if only top-k, ratio over sum of selected (common) or full — use selected sum
        NDArray ratio = new NDArray(DType.FLOAT64, k);
        for (int i = 0; i < k; i++) {
            ratio.setDouble(i, total == 0 ? 0 : explainedVar.getDouble(i) / total);
        }

        // transform: X_c @ components.T
        NDArray transformed = NPLinalg.matmul(centered, NPShape.transpose(components));
        return new Result(transformed, components, explainedVar, ratio, mean, singular);
    }

    public static Result fitTransform(NDArray X) {
        return fitTransform(X, null);
    }

    /** Project new data with a fitted result's mean/components. */
    public static NDArray transform(NDArray X, Result model) {
        if (X.shape.length != 2) throw new IllegalArgumentException("X must be 2D");
        int n = (int) X.shape[0];
        int d = (int) X.shape[1];
        NDArray mean = NPShape.reshape(model.mean, 1, d);
        NDArray centered = NPMath.subtract(X, NPShape.broadcast_to(mean, n, d));
        return NPLinalg.matmul(centered, NPShape.transpose(model.components));
    }

    /** Inverse transform from component space back to feature space. */
    public static NDArray inverseTransform(NDArray Xpca, Result model) {
        int n = (int) Xpca.shape[0];
        int d = (int) model.components.shape[1];
        NDArray rec = NPLinalg.matmul(Xpca, model.components);
        return NPMath.add(rec, NPShape.broadcast_to(NPShape.reshape(model.mean, 1, d), n, d));
    }
}
