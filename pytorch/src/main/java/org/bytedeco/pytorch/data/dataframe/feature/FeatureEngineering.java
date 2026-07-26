package org.bytedeco.pytorch.data.dataframe.feature;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.construction.Binarizer;
import org.bytedeco.pytorch.data.dataframe.feature.construction.KBinsDiscretizer;
import org.bytedeco.pytorch.data.dataframe.feature.construction.PolynomialFeatures;
import org.bytedeco.pytorch.data.dataframe.feature.encoding.LabelEncoder;
import org.bytedeco.pytorch.data.dataframe.feature.encoding.OneHotEncoder;
import org.bytedeco.pytorch.data.dataframe.feature.encoding.OrdinalEncoder;
import org.bytedeco.pytorch.data.dataframe.feature.imputation.SimpleImputer;
import org.bytedeco.pytorch.data.dataframe.feature.pipeline.DataFramePipeline;
import org.bytedeco.pytorch.data.dataframe.feature.pipeline.FeaturePipeline;
import org.bytedeco.pytorch.data.dataframe.feature.pipeline.Pipeline;
import org.bytedeco.pytorch.data.dataframe.feature.scaling.MaxAbsScaler;
import org.bytedeco.pytorch.data.dataframe.feature.scaling.MinMaxScaler;
import org.bytedeco.pytorch.data.dataframe.feature.scaling.Normalizer;
import org.bytedeco.pytorch.data.dataframe.feature.scaling.RobustScaler;
import org.bytedeco.pytorch.data.dataframe.feature.scaling.StandardScaler;
import org.bytedeco.pytorch.data.dataframe.feature.selection.VarianceThreshold;
import org.bytedeco.pytorch.data.dataframe.feature.text.TfidfVectorizer;

/**
 * Fluent feature-engineering façade bound to a {@link DataFrame}.
 *
 * <pre>
 *   DataFrame out = df.feature()
 *       .impute("mean", "age", "score")
 *       .standardScale("age", "score")
 *       .oneHot("category")
 *       .build();
 *
 *   // chain into a reusable pipeline
 *   Pipeline pipe = df.feature()
 *       .standardScale("x")
 *       .toPipeline("scale");
 * </pre>
 */
public final class FeatureEngineering {
    private DataFrame df;
    private final FeaturePipeline recorded = new FeaturePipeline();

    public FeatureEngineering(DataFrame df) {
        this.df = df;
    }

    // ---- scaling ----

    public FeatureEngineering standardScale(String... columns) throws Exception {
        StandardScaler s = new StandardScaler(columns);
        apply(s);
        return this;
    }

    public FeatureEngineering minMaxScale(String... columns) throws Exception {
        MinMaxScaler s = new MinMaxScaler(columns);
        apply(s);
        return this;
    }

    public FeatureEngineering maxAbsScale(String... columns) throws Exception {
        MaxAbsScaler s = new MaxAbsScaler(columns);
        apply(s);
        return this;
    }

    public FeatureEngineering robustScale(String... columns) throws Exception {
        RobustScaler s = new RobustScaler(columns);
        apply(s);
        return this;
    }

    public FeatureEngineering normalize(String... columns) throws Exception {
        Normalizer s = new Normalizer(columns);
        apply(s);
        return this;
    }

    // ---- encoding ----

    public FeatureEngineering oneHot(String column) throws Exception {
        OneHotEncoder enc = new OneHotEncoder(column);
        apply(enc);
        return this;
    }

    public FeatureEngineering labelEncode(String column) throws Exception {
        LabelEncoder enc = new LabelEncoder(column);
        apply(enc);
        return this;
    }

    public FeatureEngineering ordinalEncode(String... columns) throws Exception {
        OrdinalEncoder enc = new OrdinalEncoder(columns);
        apply(enc);
        return this;
    }

    // ---- imputation ----

    public FeatureEngineering impute(String strategy, String... columns) throws Exception {
        SimpleImputer imp = new SimpleImputer(strategy, columns);
        apply(imp);
        return this;
    }

    public FeatureEngineering fillMean(String... columns) throws Exception {
        return impute("mean", columns);
    }

    public FeatureEngineering fillMedian(String... columns) throws Exception {
        return impute("median", columns);
    }

    public FeatureEngineering fillMostFrequent(String... columns) throws Exception {
        return impute("most_frequent", columns);
    }

    public FeatureEngineering fillConstant(String value, String... columns) throws Exception {
        SimpleImputer imp = new SimpleImputer("constant", value, columns);
        apply(imp);
        return this;
    }

    // ---- construction / discretization ----

    public FeatureEngineering binarize(double threshold, String... columns) throws Exception {
        Binarizer b = new Binarizer(threshold, columns);
        apply(b);
        return this;
    }

    public FeatureEngineering kBins(int nBins, String... columns) throws Exception {
        KBinsDiscretizer k = new KBinsDiscretizer(nBins, "uniform", columns);
        apply(k);
        return this;
    }

    public FeatureEngineering kBins(int nBins, String strategy, String... columns) throws Exception {
        KBinsDiscretizer k = new KBinsDiscretizer(nBins, strategy, columns);
        apply(k);
        return this;
    }

    public FeatureEngineering polynomial(int degree, String... columns) throws Exception {
        PolynomialFeatures p = new PolynomialFeatures(degree, columns);
        apply(p);
        return this;
    }

    // ---- selection / text ----

    public FeatureEngineering varianceThreshold(double threshold, String... columns) throws Exception {
        VarianceThreshold v = new VarianceThreshold(threshold, columns);
        apply(v);
        return this;
    }

    public FeatureEngineering tfidf(String column) throws Exception {
        TfidfVectorizer t = new TfidfVectorizer(column);
        apply(t);
        return this;
    }

    // ---- generic / custom ----

    public FeatureEngineering transform(BaseTransformer transformer) throws Exception {
        apply(transformer);
        return this;
    }

    /** Drop rows with any nulls (DataFrame operator). */
    public FeatureEngineering dropna() throws Exception {
        df = df.dropna();
        return this;
    }

    /** Clip numeric columns. */
    public FeatureEngineering clip(Double lower, Double upper, String... columns) throws Exception {
        df = df.clip(lower, upper, columns);
        return this;
    }

    // ---- output ----

    public DataFrame build() {
        return df;
    }

    public DataFrame get() {
        return df;
    }

    /** Export recorded transformers as a reusable {@link FeaturePipeline}. */
    public FeaturePipeline toFeaturePipeline() {
        return recorded;
    }

    /** Export as sklearn-style {@link Pipeline} with auto step names. */
    public Pipeline toPipeline() {
        Pipeline p = new Pipeline();
        int i = 0;
        for (BaseTransformer t : recorded.getTransformers()) {
            p.addStep("step_" + (i++), t);
        }
        return p;
    }

    public Pipeline toPipeline(String... stepNames) {
        Pipeline p = new Pipeline();
        java.util.List<BaseTransformer> ts = recorded.getTransformers();
        for (int i = 0; i < ts.size(); i++) {
            String name = (stepNames != null && i < stepNames.length) ? stepNames[i] : "step_" + i;
            p.addStep(name, ts.get(i));
        }
        return p;
    }

    /** Export as bound {@link DataFramePipeline}. */
    public DataFramePipeline toDataFramePipeline() {
        DataFramePipeline p = new DataFramePipeline(df);
        int i = 0;
        for (BaseTransformer t : recorded.getTransformers()) {
            p.append("step_" + (i++), t);
        }
        return p;
    }

    private void apply(BaseTransformer t) throws Exception {
        recorded.addTransformer(t);
        df = t.fitTransform(df);
    }
}
