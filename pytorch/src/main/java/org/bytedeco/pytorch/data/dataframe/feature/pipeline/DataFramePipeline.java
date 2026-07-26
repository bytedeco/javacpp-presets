package org.bytedeco.pytorch.data.dataframe.feature.pipeline;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseEstimator;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;

import java.io.*;
import java.util.*;
import java.util.function.Function;

/**
 * Spark-MLlib / sklearn style chainable pipeline bound to a {@link DataFrame}.
 *
 * <pre>
 *   DataFrame out = df.pipeline()
 *       .append("impute", new SimpleImputer("mean", "age", "score"))
 *       .append("scale",  new StandardScaler("age", "score"))
 *       .append("ohe",    new OneHotEncoder("category"))
 *       .fitTransform();
 *
 *   // with estimator
 *   DataFramePipeline pipe = df.pipeline()
 *       .append("scale", new StandardScaler("x1","x2"))
 *       .append("clf",   new LogisticRegression());
 *   pipe.fit(new String[]{"x1","x2"}, "y");
 *   double[] preds = pipe.predict(testDf, new String[]{"x1","x2"});
 * </pre>
 *
 * <p>Stages can be {@link BaseTransformer}, {@link BaseClassifier},
 * {@link BaseRegressor}, or a lambda {@code Function&lt;DataFrame,DataFrame&gt;}.
 */
public final class DataFramePipeline implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<String> names = new ArrayList<>();
    private final List<Object> stages = new ArrayList<>();
    private transient DataFrame source;
    private DataFrame lastTransformed;
    private boolean fitted = false;
    private String[] featureCols;
    private String labelCol;

    public DataFramePipeline(DataFrame source) {
        this.source = source;
    }

    public DataFramePipeline() {
        this.source = null;
    }

    // ---- builder ----

    public DataFramePipeline append(String name, BaseTransformer transformer) {
        names.add(name);
        stages.add(transformer);
        return this;
    }

    public DataFramePipeline append(String name, BaseClassifier classifier) {
        names.add(name);
        stages.add(classifier);
        return this;
    }

    public DataFramePipeline append(String name, BaseRegressor regressor) {
        names.add(name);
        stages.add(regressor);
        return this;
    }

    /** Stateless functional stage (not serialized as lambda unless you use a named class). */
    public DataFramePipeline append(String name, Function<DataFrame, DataFrame> fn) {
        names.add(name);
        stages.add(fn);
        return this;
    }

    public DataFramePipeline append(String name, PipelineStage stage) {
        names.add(name);
        stages.add(stage);
        return this;
    }

    /** Alias for {@link #append(String, BaseTransformer)}. */
    public DataFramePipeline addStage(String name, BaseTransformer t) { return append(name, t); }
    public DataFramePipeline addStage(String name, BaseClassifier c) { return append(name, c); }
    public DataFramePipeline addStage(String name, BaseRegressor r) { return append(name, r); }

    public DataFramePipeline setDataFrame(DataFrame df) {
        this.source = df;
        return this;
    }

    public DataFrame getDataFrame() { return source; }

    // ---- fit / transform ----

    /** Fit transformer stages on the bound DataFrame (no final estimator). */
    public DataFramePipeline fit() throws Exception {
        requireSource();
        return fit(source);
    }

    public DataFramePipeline fit(DataFrame df) throws Exception {
        this.source = df;
        DataFrame cur = df;
        for (Object stage : stages) {
            if (stage instanceof BaseTransformer t) {
                t.fit(cur);
                cur = t.transform(cur);
            } else if (stage instanceof PipelineStage ps) {
                ps.fit(cur);
                cur = ps.transform(cur);
            } else if (stage instanceof Function) {
                @SuppressWarnings("unchecked")
                Function<DataFrame, DataFrame> fn = (Function<DataFrame, DataFrame>) stage;
                cur = fn.apply(cur);
            }
            // classifiers/regressors need feature/label — skip here
        }
        this.lastTransformed = cur;
        this.fitted = true;
        return this;
    }

    /** Fit full pipeline including final estimator. */
    public DataFramePipeline fit(String[] featureCols, String labelCol) throws Exception {
        requireSource();
        return fit(source, featureCols, labelCol);
    }

    public DataFramePipeline fit(DataFrame df, String[] featureCols, String labelCol) throws Exception {
        this.source = df;
        this.featureCols = featureCols;
        this.labelCol = labelCol;
        DataFrame cur = df;
        int last = stages.size() - 1;
        for (int i = 0; i < stages.size(); i++) {
            Object stage = stages.get(i);
            if (stage instanceof BaseTransformer t) {
                t.fit(cur);
                cur = t.transform(cur);
            } else if (stage instanceof PipelineStage ps) {
                if (i == last && ps.isEstimator()) {
                    ps.fit(cur, featureCols, labelCol);
                } else {
                    ps.fit(cur);
                    cur = ps.transform(cur);
                }
            } else if (stage instanceof Function) {
                @SuppressWarnings("unchecked")
                Function<DataFrame, DataFrame> fn = (Function<DataFrame, DataFrame>) stage;
                cur = fn.apply(cur);
            } else if (i == last) {
                if (stage instanceof BaseClassifier clf) clf.fit(cur, featureCols, labelCol);
                else if (stage instanceof BaseRegressor reg) reg.fit(cur, featureCols, labelCol);
            }
        }
        this.lastTransformed = cur;
        this.fitted = true;
        return this;
    }

    public DataFrame transform() throws Exception {
        requireSource();
        return transform(source);
    }

    public DataFrame transform(DataFrame df) throws Exception {
        DataFrame cur = df;
        for (Object stage : stages) {
            if (stage instanceof BaseTransformer t) {
                if (!t.isFitted()) throw new IllegalStateException(t.getClass().getSimpleName() + " not fitted");
                cur = t.transform(cur);
            } else if (stage instanceof PipelineStage ps) {
                cur = ps.transform(cur);
            } else if (stage instanceof Function) {
                @SuppressWarnings("unchecked")
                Function<DataFrame, DataFrame> fn = (Function<DataFrame, DataFrame>) stage;
                cur = fn.apply(cur);
            }
            // skip final estimator
        }
        this.lastTransformed = cur;
        return cur;
    }

    public DataFrame fitTransform() throws Exception {
        return fit().transform();
    }

    public DataFrame fitTransform(DataFrame df) throws Exception {
        return fit(df).transform(df);
    }

    // ---- predict ----

    public double[] predict(DataFrame df, String[] featureCols) throws Exception {
        DataFrame cur = df;
        for (Object stage : stages) {
            if (stage instanceof BaseTransformer t) {
                cur = t.transform(cur);
            } else if (stage instanceof PipelineStage ps && !ps.isEstimator()) {
                cur = ps.transform(cur);
            } else if (stage instanceof Function) {
                @SuppressWarnings("unchecked")
                Function<DataFrame, DataFrame> fn = (Function<DataFrame, DataFrame>) stage;
                cur = fn.apply(cur);
            } else if (stage instanceof BaseClassifier clf) {
                return clf.predict(BaseClassifier.extractMatrix(cur, featureCols));
            } else if (stage instanceof BaseRegressor reg) {
                return reg.predict(reg.extractMatrix(cur, featureCols));
            } else if (stage instanceof PipelineStage ps && ps.isEstimator()) {
                return ps.predict(cur, featureCols);
            }
        }
        throw new IllegalStateException("No estimator in pipeline");
    }

    public double[] predict(String[] featureCols) throws Exception {
        requireSource();
        return predict(source, featureCols);
    }

    public double score(DataFrame df, String[] featureCols, String labelCol) throws Exception {
        DataFrame cur = df;
        for (Object stage : stages) {
            if (stage instanceof BaseTransformer t) cur = t.transform(cur);
            else if (stage instanceof Function) {
                @SuppressWarnings("unchecked")
                Function<DataFrame, DataFrame> fn = (Function<DataFrame, DataFrame>) stage;
                cur = fn.apply(cur);
            } else if (stage instanceof BaseClassifier clf) {
                double[][] X = BaseClassifier.extractMatrix(cur, featureCols);
                double[] y = extractLabel(cur, labelCol);
                return clf.score(X, y);
            } else if (stage instanceof BaseRegressor reg) {
                double[][] X = reg.extractMatrix(cur, featureCols);
                double[] y = extractLabel(cur, labelCol);
                return reg.score(X, y);
            }
        }
        throw new IllegalStateException("No estimator in pipeline");
    }

    // ---- introspection ----

    public boolean isFitted() { return fitted; }
    public List<String> stageNames() { return Collections.unmodifiableList(names); }
    public Object getStage(String name) {
        int i = names.indexOf(name);
        if (i < 0) throw new IllegalArgumentException("No stage: " + name);
        return stages.get(i);
    }
    public DataFrame getLastTransformed() { return lastTransformed; }
    public Pipeline toPipeline() {
        Pipeline p = new Pipeline();
        for (int i = 0; i < stages.size(); i++) {
            Object s = stages.get(i);
            String n = names.get(i);
            if (s instanceof BaseTransformer t) p.addStep(n, t);
            else if (s instanceof BaseClassifier c) p.addStep(n, c);
            else if (s instanceof BaseRegressor r) p.addStep(n, r);
        }
        return p;
    }

    public void save(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        }
    }

    public static DataFramePipeline load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return (DataFramePipeline) ois.readObject();
        }
    }

    // ---- helpers ----

    private void requireSource() {
        if (source == null) throw new IllegalStateException("No DataFrame bound; call setDataFrame() or fit(df)");
    }

    private static double[] extractLabel(DataFrame df, String labelCol) {
        double[] y = new double[df.rowCount()];
        for (int i = 0; i < y.length; i++) {
            Object v = df.get(i, labelCol);
            y[i] = v instanceof Number ? ((Number) v).doubleValue() : Double.NaN;
        }
        return y;
    }

    /**
     * Optional unified stage interface (Spark-like) for custom stages.
     */
    public interface PipelineStage extends Serializable {
        default void fit(DataFrame df) throws Exception {}
        default void fit(DataFrame df, String[] featureCols, String labelCol) throws Exception {
            fit(df);
        }
        DataFrame transform(DataFrame df) throws Exception;
        default boolean isEstimator() { return false; }
        default double[] predict(DataFrame df, String[] featureCols) throws Exception {
            throw new UnsupportedOperationException("not an estimator");
        }
    }
}
