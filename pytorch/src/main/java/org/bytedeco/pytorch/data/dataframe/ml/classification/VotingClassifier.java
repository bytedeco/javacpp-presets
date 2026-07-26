package org.bytedeco.pytorch.data.dataframe.ml.classification;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * 投票分类器（软投票 / 硬投票）
 */
public class VotingClassifier extends BaseClassifier {
    private String voting; // "hard" | "soft"
    private List<String> names = new ArrayList<>();
    private List<BaseClassifier> estimators = new ArrayList<>();
    private double[] classes;

    public VotingClassifier(String voting) { this.voting = voting; }

    public VotingClassifier addEstimator(String name, BaseClassifier clf) {
        names.add(name); estimators.add(clf); return this;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        TreeSet<Double> cs = new TreeSet<>(); for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();
        for (BaseClassifier e : estimators) e.fit(X, y);
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        if ("soft".equals(voting)) {
            double[][] scores = new double[X.length][classes.length];
            for (BaseClassifier e : estimators) {
                double[][] proba = e.predictProba(X);
                for (int i = 0; i < X.length; i++)
                    for (int c = 0; c < classes.length; c++) scores[i][c] += proba[i][c];
            }
            double[] result = new double[X.length];
            for (int i = 0; i < X.length; i++) {
                int best = 0; for (int c = 1; c < classes.length; c++) if (scores[i][c] > scores[i][best]) best = c;
                result[i] = classes[best];
            }
            return result;
        } else { // hard
            double[][] votes = new double[X.length][classes.length];
            for (BaseClassifier e : estimators) {
                double[] preds = e.predict(X);
                for (int i = 0; i < preds.length; i++)
                    for (int c = 0; c < classes.length; c++)
                        if (preds[i] == classes[c]) { votes[i][c]++; break; }
            }
            double[] result = new double[X.length];
            for (int i = 0; i < X.length; i++) {
                int best = 0; for (int c = 1; c < classes.length; c++) if (votes[i][c] > votes[i][best]) best = c;
                result[i] = classes[best];
            }
            return result;
        }
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("voting", voting); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("voting")) voting = (String) params.get("voting");
    }
}

