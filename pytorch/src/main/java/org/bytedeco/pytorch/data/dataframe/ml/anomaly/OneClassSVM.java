package org.bytedeco.pytorch.data.dataframe.ml.anomaly;

import org.bytedeco.pytorch.data.dataframe.ml.classification.SGDClassifier;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import java.util.*;

/** 单类 SVM（One-Class SVM，使用 RBF 核近似） */
public class OneClassSVM extends BaseClassifier {
    private double nu; private String kernel; private double gamma;
    private double[][] supportVectors; private double[] alphas; private double rho;
    private double threshold;

    public OneClassSVM() { this(0.5, "rbf", -1); }
    public OneClassSVM(double nu, String kernel, double gamma) { this.nu = nu; this.kernel = kernel; this.gamma = gamma; }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        // Simplified: use distance-to-centroid with kernel as approximation
        int n = X.length, d = X[0].length;
        double g = gamma < 0 ? 1.0 / d : gamma;
        // Compute kernel mean map (Parzen density estimator)
        supportVectors = X; alphas = new double[n]; Arrays.fill(alphas, 1.0 / n);
        // Compute scores on training data, use nu-percentile as threshold
        double[] scores = scoreAll(X, g);
        Arrays.sort(scores.clone()); // don't sort the original
        int nuIdx = (int)(nu * n); nuIdx = Math.max(0, Math.min(nuIdx, n-1));
        double[] sorted = scores.clone(); Arrays.sort(sorted);
        rho = sorted[nuIdx]; threshold = rho;
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double g = gamma < 0 ? 1.0 / supportVectors[0].length : gamma;
        double[] scores = scoreAll(X, g);
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) result[i] = scores[i] >= threshold ? 1.0 : -1.0;
        return result;
    }

    private double[] scoreAll(double[][] X, double g) {
        double[] s = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < supportVectors.length; j++)
                s[i] += alphas[j] * kernelFunc(X[i], supportVectors[j], g);
        }
        return s;
    }

    private double kernelFunc(double[] a, double[] b, double g) {
        if ("linear".equals(kernel)) { double s=0; for (int i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
        double s = 0; for (int i = 0; i < a.length; i++) s += (a[i]-b[i])*(a[i]-b[i]);
        return Math.exp(-g * s);
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("nu", nu); p.put("kernel", kernel); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("nu")) nu = ((Number) params.get("nu")).doubleValue();
    }
}

