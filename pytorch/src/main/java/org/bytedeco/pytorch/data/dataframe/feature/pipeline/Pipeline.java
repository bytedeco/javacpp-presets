package org.bytedeco.pytorch.data.dataframe.feature.pipeline;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseEstimator;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;

import java.io.*;
import java.util.*;

/**
 * sklearn 风格 Pipeline：串联 Transformer 步骤 + 末尾可选 Estimator
 *
 * <pre>
 * Pipeline pipe = new Pipeline()
 *     .addStep("scaler", new StandardScaler("age","score"))
 *     .addStep("pca",    new PCA(2))
 *     .addStep("clf",    new LogisticRegression());
 *
 * pipe.fit(trainDf, featureCols, "label");
 * double[] preds = pipe.predict(testMatrix);
 * </pre>
 */
public class Pipeline implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<String> names = new ArrayList<>();
    private final List<Object> steps = new ArrayList<>();  // BaseTransformer or BaseClassifier/BaseRegressor

    // ---------- builder ----------

    public Pipeline addStep(String name, BaseTransformer transformer) {
        names.add(name);
        steps.add(transformer);
        return this;
    }

    public Pipeline addStep(String name, BaseClassifier classifier) {
        names.add(name);
        steps.add(classifier);
        return this;
    }

    public Pipeline addStep(String name, BaseRegressor regressor) {
        names.add(name);
        steps.add(regressor);
        return this;
    }

    // convenience
    public static Pipeline make(Object... nameStepPairs) {
        Pipeline p = new Pipeline();
        for (int i = 0; i < nameStepPairs.length - 1; i += 2) {
            String n = (String) nameStepPairs[i];
            Object s = nameStepPairs[i + 1];
            if (s instanceof BaseTransformer) p.addStep(n, (BaseTransformer) s);
            else if (s instanceof BaseClassifier) p.addStep(n, (BaseClassifier) s);
            else if (s instanceof BaseRegressor)  p.addStep(n, (BaseRegressor)  s);
            else throw new IllegalArgumentException("Unknown step type: " + s.getClass());
        }
        return p;
    }

    // ---------- fit / transform with DataFrame ----------

    public Pipeline fit(DataFrame df) throws Exception {
        DataFrame cur = df;
        for (Object step : steps) {
            if (step instanceof BaseTransformer t) {
                t.fit(cur);
                cur = t.transform(cur);
            }
            // final estimator fitted separately via fit(df, featureCols, labelCol)
        }
        return this;
    }

    public Pipeline fit(DataFrame df, String[] featureCols, String labelCol) throws Exception {
        DataFrame cur = df;
        int last = steps.size() - 1;
        for (int i = 0; i < steps.size(); i++) {
            Object step = steps.get(i);
            if (step instanceof BaseTransformer t) {
                t.fit(cur);
                cur = t.transform(cur);
            } else if (i == last) {
                if (step instanceof BaseClassifier clf) clf.fit(cur, featureCols, labelCol);
                else if (step instanceof BaseRegressor reg) reg.fit(cur, featureCols, labelCol);
            }
        }
        return this;
    }

    /** fit on double[][] (only transforms first, then final estimator) */
    public Pipeline fit(double[][] X, double[] y) {
        int last = steps.size() - 1;
        double[][] cur = X;
        for (int i = 0; i < steps.size(); i++) {
            Object step = steps.get(i);
            if (i == last) {
                if (step instanceof BaseClassifier clf) clf.fit(cur, y);
                else if (step instanceof BaseRegressor reg) reg.fit(cur, y);
            }
            // Note: BaseTransformer doesn't support double[][] natively; skip for raw path
        }
        return this;
    }

    public DataFrame transform(DataFrame df) throws Exception {
        DataFrame cur = df;
        for (Object step : steps) {
            if (step instanceof BaseTransformer t) {
                if (!t.isFitted()) throw new IllegalStateException(t.getClass().getSimpleName() + " not fitted");
                cur = t.transform(cur);
            }
        }
        return cur;
    }

    public DataFrame fitTransform(DataFrame df) throws Exception {
        return fit(df).transform(df);
    }

    // ---------- predict ----------

    public double[] predict(double[][] X) {
        double[][] cur = X;
        for (Object step : steps) {
            if (step instanceof BaseClassifier clf) return clf.predict(cur);
            if (step instanceof BaseRegressor reg)  return reg.predict(cur);
        }
        throw new IllegalStateException("No estimator in pipeline");
    }

    public double[] predict(DataFrame df, String[] featureCols) throws Exception {
        DataFrame cur = df;
        for (Object step : steps) {
            if (step instanceof BaseTransformer t) cur = t.transform(cur);
            else if (step instanceof BaseClassifier clf) {
                double[][] mat = clf.extractMatrix(cur, featureCols);
                return clf.predict(mat);
            } else if (step instanceof BaseRegressor reg) {
                double[][] mat = reg.extractMatrix(cur, featureCols);
                return reg.predict(mat);
            }
        }
        throw new IllegalStateException("No estimator in pipeline");
    }

    public double[][] predictProba(double[][] X) {
        for (Object step : steps) {
            if (step instanceof BaseClassifier clf) return clf.predictProba(X);
        }
        throw new IllegalStateException("No classifier in pipeline");
    }

    public double score(double[][] X, double[] y) {
        for (Object step : steps) {
            if (step instanceof BaseClassifier clf) return clf.score(X, y);
            if (step instanceof BaseRegressor reg)  return reg.score(X, y);
        }
        throw new IllegalStateException("No estimator in pipeline");
    }

    // ---------- param access ----------

    public Object getNamedStep(String name) {
        int idx = names.indexOf(name);
        if (idx < 0) throw new IllegalArgumentException("No step named: " + name);
        return steps.get(idx);
    }

    public List<String> getStepNames() { return Collections.unmodifiableList(names); }

    /** set_params(step__param=value) style */
    public void setParam(String stepName, String paramName, Object value) {
        Object step = getNamedStep(stepName);
        if (step instanceof BaseEstimator est) {
            Map<String, Object> params = new HashMap<>();
            params.put(paramName, value);
            est.setParams(params);
        }
    }

    // ---------- serialization ----------

    public void save(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        }
    }

    public static Pipeline load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return (Pipeline) ois.readObject();
        }
    }
}

