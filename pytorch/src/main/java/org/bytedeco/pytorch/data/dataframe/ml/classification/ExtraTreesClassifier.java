package org.bytedeco.pytorch.data.dataframe.ml.classification;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * Extra Trees 分类器（极端随机化树）
 * 与 RandomForest 的区别：分裂阈值完全随机选取（不搜索最优）
 */
public class ExtraTreesClassifier extends BaseClassifier {
    private int nEstimators;
    private Integer maxDepth;
    private int minSamplesSplit;
    private Long randomState;

    private List<ExtraTreeClassifier> trees = new ArrayList<>();
    private double[] classes;

    public ExtraTreesClassifier() { this(100, null, 2, null); }
    public ExtraTreesClassifier(int nEstimators, Integer maxDepth, int minSamplesSplit, Long randomState) {
        this.nEstimators = nEstimators; this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit; this.randomState = randomState;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        TreeSet<Double> cs = new TreeSet<>(); for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();
        trees.clear();
        Random rng = randomState == null ? new Random() : new Random(randomState);
        for (int t = 0; t < nEstimators; t++) {
            ExtraTreeClassifier tree = new ExtraTreeClassifier(maxDepth, minSamplesSplit,
                randomState == null ? null : randomState + t);
            tree.fit(X, y);
            trees.add(tree);
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[][] votes = new double[X.length][classes.length];
        for (ExtraTreeClassifier tree : trees) {
            double[] preds = tree.predict(X);
            for (int i = 0; i < preds.length; i++)
                for (int c = 0; c < classes.length; c++)
                    if (preds[i] == classes[c]) { votes[i][c]++; break; }
        }
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            int best = 0;
            for (int c = 1; c < classes.length; c++) if (votes[i][c] > votes[i][best]) best = c;
            result[i] = classes[best];
        }
        return result;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators); p.put("max_depth", maxDepth); p.put("random_state", randomState); return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_estimators")) nEstimators = ((Number) params.get("n_estimators")).intValue();
    }
}

