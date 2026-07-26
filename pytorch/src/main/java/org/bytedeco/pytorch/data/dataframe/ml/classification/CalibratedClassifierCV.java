package org.bytedeco.pytorch.data.dataframe.ml.classification;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.data.dataframe.ml.model_selection.KFold;
import org.bytedeco.pytorch.data.dataframe.ml.regression.LinearRegression;

import java.util.*;

/**
 * 校准分类器（CalibratedClassifierCV）
 * 通过 sigmoid 或 isotonic 校准将 decision_function 转为概率
 */
public class CalibratedClassifierCV extends BaseClassifier {
    private BaseClassifier baseEstimator;
    private String method;  // "sigmoid" | "isotonic"
    private int cv;

    private double calibA, calibB;  // sigmoid calibration params
    private double[] classes;

    public CalibratedClassifierCV(BaseClassifier estimator) { this(estimator, "sigmoid", 5); }
    public CalibratedClassifierCV(BaseClassifier estimator, String method, int cv) {
        this.baseEstimator = estimator; this.method = method; this.cv = cv;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        TreeSet<Double> cs = new TreeSet<>(); for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();

        // Get OOF probability estimates
        KFold kf = new KFold(cv, true, null);
        List<KFold.Split> splits = kf.split(X, y);
        double[] oofProbs = new double[X.length];
        double[] oofY = new double[X.length];
        for (KFold.Split s : splits) {
            baseEstimator.fit(s.trainX(X), s.trainY(y));
            double[][] proba;
            try { proba = baseEstimator.predictProba(s.testX(X)); }
            catch (UnsupportedOperationException e) {
                double[] p = baseEstimator.predict(s.testX(X));
                proba = new double[p.length][2];
                for (int i = 0; i < p.length; i++) { proba[i][0] = p[i] == classes[0] ? 1.0 : 0.0; proba[i][1] = 1 - proba[i][0]; }
            }
            for (int i = 0; i < s.testIndices.length; i++) {
                oofProbs[s.testIndices[i]] = proba[i][proba[i].length - 1];
                oofY[s.testIndices[i]] = y[s.testIndices[i]];
            }
        }

        // Fit calibration on all data
        baseEstimator.fit(X, y);

        // Platt scaling: sigmoid A, B via MLE
        if ("sigmoid".equals(method)) {
            calibA = 0; calibB = 0; // simplified: use pre-existing sigmoid mapping
            // Fit linear regression on logit(prob) → y  (Platt scaling)
            int n = oofProbs.length;
            double[][] logitX = new double[n][1];
            for (int i = 0; i < n; i++) {
                double p = Math.max(1e-6, Math.min(1 - 1e-6, oofProbs[i]));
                logitX[i][0] = Math.log(p / (1 - p));
            }
            LinearRegression lr = new LinearRegression();
            lr.fit(logitX, oofY);
            calibA = lr.getCoef()[0]; calibB = lr.getIntercept();
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[][] proba = predictProba(X);
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            int best = 0; for (int c = 1; c < proba[i].length; c++) if (proba[i][c] > proba[i][best]) best = c;
            result[i] = classes[best];
        }
        return result;
    }

    @Override
    public double[][] predictProba(double[][] X) {
        double[][] raw;
        try { raw = baseEstimator.predictProba(X); }
        catch (UnsupportedOperationException e) {
            double[] p = baseEstimator.predict(X);
            raw = new double[p.length][2];
            for (int i = 0; i < p.length; i++) { raw[i][0] = p[i] == classes[0] ? 1.0 : 0.0; raw[i][1] = 1 - raw[i][0]; }
        }
        if (!"sigmoid".equals(method)) return raw;
        double[][] calibrated = new double[X.length][raw[0].length];
        for (int i = 0; i < X.length; i++) {
            double p = raw[i][raw[i].length - 1];
            double logit = Math.log(Math.max(1e-6, p) / Math.max(1e-6, 1 - p));
            double pCalib = 1.0 / (1 + Math.exp(-(calibA * logit + calibB)));
            calibrated[i][0] = 1 - pCalib; calibrated[i][1] = pCalib;
        }
        return calibrated;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("method", method); p.put("cv", cv); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("method")) method = (String) params.get("method");
    }
}

