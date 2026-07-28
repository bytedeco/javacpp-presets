package org.bytedeco.pytorch.dataframe.feature.pipeline;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.dataframe.feature.base.BaseEstimator;
import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;

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
        return fitInternal(df, null, null);
    }

    /**
     * sklearn-style supervised fit: intermediate transformers that accept y
     * (SelectKBest, SelectPercentile, TargetEncoder, RFE, …) receive {@code y}.
     */
    public Pipeline fit(DataFrame df, double[] y) throws Exception {
        return fitInternal(df, y, null);
    }

    /**
     * Supervised fit using a label column already present in {@code df}.
     * Also used to fit a final classifier/regressor step.
     */
    public Pipeline fit(DataFrame df, String labelCol) throws Exception {
        return fitInternal(df, null, labelCol);
    }

    public Pipeline fit(DataFrame df, String[] featureCols, String labelCol) throws Exception {
        DataFrame cur = df;
        int last = steps.size() - 1;
        double[] y = extractY(df, labelCol);
        for (int i = 0; i < steps.size(); i++) {
            Object step = steps.get(i);
            if (step instanceof BaseTransformer t) {
                fitTransformer(t, cur, y, labelCol);
                cur = t.transform(cur);
            } else if (i == last) {
                if (step instanceof BaseClassifier clf) clf.fit(cur, featureCols, labelCol);
                else if (step instanceof BaseRegressor reg) reg.fit(cur, featureCols, labelCol);
            }
        }
        return this;
    }

    private Pipeline fitInternal(DataFrame df, double[] y, String labelCol) throws Exception {
        DataFrame cur = df;
        if (y == null && labelCol != null && df.hasColumn(labelCol)) {
            y = extractY(df, labelCol);
        }
        int last = steps.size() - 1;
        for (int i = 0; i < steps.size(); i++) {
            Object step = steps.get(i);
            if (step instanceof BaseTransformer t) {
                fitTransformer(t, cur, y, labelCol);
                cur = t.transform(cur);
            } else if (i == last && labelCol != null) {
                // final estimator with label column — use all current numeric-ish cols except label
                List<String> feats = new ArrayList<>();
                for (var c : cur.columns()) {
                    if (!c.name().equals(labelCol)) feats.add(c.name());
                }
                String[] featureCols = feats.toArray(new String[0]);
                if (step instanceof BaseClassifier clf) clf.fit(cur, featureCols, labelCol);
                else if (step instanceof BaseRegressor reg) reg.fit(cur, featureCols, labelCol);
            }
        }
        return this;
    }

    /**
     * Fit a transformer, preferring supervised overloads when y/labelCol is available:
     * <ol>
     *   <li>{@code fit(DataFrame, double[])}</li>
     *   <li>{@code fit(DataFrame, String labelCol)} / {@code setLabelCol}+{@code fit}</li>
     *   <li>plain {@code fit(DataFrame)}</li>
     * </ol>
     */
    private static void fitTransformer(BaseTransformer t, DataFrame cur, double[] y, String labelCol)
            throws Exception {
        // 1) fit(DataFrame, double[])
        if (y != null) {
            try {
                var m = t.getClass().getMethod("fit", DataFrame.class, double[].class);
                m.invoke(t, cur, y);
                return;
            } catch (NoSuchMethodException ignored) {
            } catch (java.lang.reflect.InvocationTargetException e) {
                Throwable c = e.getCause() != null ? e.getCause() : e;
                if (c instanceof Exception ex) throw ex;
                throw new RuntimeException(c);
            }
        }
        // 2) setLabelCol + fit, or fit(DataFrame, String)
        if (labelCol != null) {
            try {
                var m = t.getClass().getMethod("fit", DataFrame.class, String.class);
                m.invoke(t, cur, labelCol);
                return;
            } catch (NoSuchMethodException ignored) {
            } catch (java.lang.reflect.InvocationTargetException e) {
                Throwable c = e.getCause() != null ? e.getCause() : e;
                if (c instanceof Exception ex) throw ex;
                throw new RuntimeException(c);
            }
            try {
                var set = t.getClass().getMethod("setLabelCol", String.class);
                set.invoke(t, labelCol);
            } catch (NoSuchMethodException ignored) {}
        }
        // 3) plain fit
        t.fit(cur);
    }

    private static double[] extractY(DataFrame df, String labelCol) {
        if (labelCol == null || !df.hasColumn(labelCol)) {
            throw new IllegalArgumentException("label column missing: " + labelCol);
        }
        int n = df.rowCount();
        double[] y = new double[n];
        var col = df.column(labelCol);
        for (int i = 0; i < n; i++) {
            Object v = col.get(i);
            y[i] = v == null ? 0.0 : org.bytedeco.pytorch.dataframe.DataValues.asDouble(v);
        }
        return y;
    }

    /** fit on double[][] (only final estimator; transformers need DataFrame path) */
    public Pipeline fit(double[][] X, double[] y) {
        int last = steps.size() - 1;
        double[][] cur = X;
        for (int i = 0; i < steps.size(); i++) {
            Object step = steps.get(i);
            if (i == last) {
                if (step instanceof BaseClassifier clf) clf.fit(cur, y);
                else if (step instanceof BaseRegressor reg) reg.fit(cur, y);
            }
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

    /** Supervised fit_transform (for pipelines ending in SelectKBest / similar). */
    public DataFrame fitTransform(DataFrame df, double[] y) throws Exception {
        fit(df, y);
        return transform(df);
    }

    public DataFrame fitTransform(DataFrame df, String labelCol) throws Exception {
        fit(df, labelCol);
        return transform(df);
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

