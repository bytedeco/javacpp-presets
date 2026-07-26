package org.bytedeco.pytorch.data.dataframe.feature.pipeline;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.io.Serializable;
import java.util.*;

/**
 * FeatureUnion：并行运行多个 Transformer 并拼接结果（对应 sklearn FeatureUnion）
 *
 * <pre>
 * FeatureUnion union = new FeatureUnion()
 *     .addTransformer("pca",  new PCA(3))
 *     .addTransformer("kbest", new SelectKBest(5, "f_score", "f1","f2","f3"));
 *
 * DataFrame result = union.fitTransform(df);
 * </pre>
 */
public class FeatureUnion extends BaseTransformer implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<String> names = new ArrayList<>();
    private final List<BaseTransformer> transformers = new ArrayList<>();

    public FeatureUnion() { super(); }

    public FeatureUnion addTransformer(String name, BaseTransformer t) {
        names.add(name);
        transformers.add(t);
        return this;
    }

    public static FeatureUnion make(Object... nameTransformerPairs) {
        FeatureUnion fu = new FeatureUnion();
        for (int i = 0; i < nameTransformerPairs.length - 1; i += 2) {
            fu.addTransformer((String) nameTransformerPairs[i], (BaseTransformer) nameTransformerPairs[i + 1]);
        }
        return fu;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (BaseTransformer t : transformers) t.fit(X);
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("FeatureUnion not fitted");
        DataFrame result = null;
        for (int i = 0; i < transformers.size(); i++) {
            BaseTransformer t = transformers.get(i);
            DataFrame part = t.transform(X);
            String prefix = names.get(i) + "__";
            if (result == null) {
                result = renameCols(part, prefix);
            } else {
                DataFrame renamed = renameCols(part, prefix);
                for (String col : renamed.getColumnNames()) {
                    result = result.withColumn(col, renamed.column(col).data());
                }
            }
        }
        return result != null ? result : X;
    }

    @Override
    public DataFrame fitTransform(DataFrame X) throws Exception {
        fit(X);
        return transform(X);
    }

    private DataFrame renameCols(DataFrame df, String prefix) throws Exception {
        DataFrame result = DataFrame.create();
        boolean first = true;
        for (String col : df.getColumnNames()) {
            String newName = prefix + col;
            if (first) {
                result = df.select(col);
                // rename the single column – workaround: use withColumn on a fresh df
                result = DataFrame.create();
                result = result.withColumn(newName, df.column(col).data());
                first = false;
            } else {
                result = result.withColumn(newName, df.column(col).data());
            }
        }
        return result;
    }

    public List<String> getTransformerNames() { return Collections.unmodifiableList(names); }
}

