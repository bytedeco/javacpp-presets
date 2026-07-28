package org.bytedeco.pytorch.dataframe.feature.pipeline;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.io.Serializable;
import java.util.*;

/**
 * ColumnTransformer：对不同列子集应用不同的 Transformer（对应 sklearn ColumnTransformer）
 *
 * <pre>
 * ColumnTransformer ct = new ColumnTransformer()
 *     .addTransformer("num", new StandardScaler(), "age", "income")
 *     .addTransformer("cat", new OneHotEncoder(), "gender", "city")
 *     .setRemainder("passthrough");
 *
 * DataFrame result = ct.fitTransform(df);
 * </pre>
 */
public class ColumnTransformer extends BaseTransformer implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<String> names = new ArrayList<>();
    private final List<BaseTransformer> transformers = new ArrayList<>();
    private final List<List<String>> columnSets = new ArrayList<>();
    private String remainder = "drop"; // "drop" | "passthrough"

    public ColumnTransformer() { super(); }

    public ColumnTransformer addTransformer(String name, BaseTransformer t, String... cols) {
        names.add(name);
        transformers.add(t);
        columnSets.add(Arrays.asList(cols));
        return this;
    }

    public ColumnTransformer setRemainder(String remainder) {
        this.remainder = remainder;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        // sklearn semantics: each transformer is fit ONLY on its column subset
        for (int i = 0; i < transformers.size(); i++) {
            BaseTransformer t = transformers.get(i);
            List<String> cols = columnSets.get(i);
            DataFrame sub = X.select(cols.toArray(new String[0]));
            t.fit(sub);
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("ColumnTransformer not fitted");

        // Collect all handled columns
        Set<String> handled = new HashSet<>();
        for (List<String> cols : columnSets) handled.addAll(cols);

        // Start with remainder columns if passthrough
        DataFrame result = DataFrame.create();

        // Apply each transformer and collect new columns
        for (int i = 0; i < transformers.size(); i++) {
            BaseTransformer t = transformers.get(i);
            List<String> cols = columnSets.get(i);

            // Select only the relevant columns
            DataFrame sub = X.select(cols.toArray(new String[0]));
            DataFrame transformed = t.transform(sub);

            // Append all columns from transformed to result
            List<String> tCols = transformed.getColumnNames();
            for (String col : tCols) {
                List<?> data = transformed.column(col).data();
                // Add rows to result if empty, else append columns
                if (result.rowCount() == 0 && transformed.rowCount() > 0) {
                    result = transformed.select(col);
                } else {
                    result = result.withColumn(col, data);
                }
            }
        }

        // Handle remainder
        if ("passthrough".equals(remainder)) {
            List<String> allCols = X.getColumnNames();
            for (String col : allCols) {
                if (!handled.contains(col)) {
                    List<?> data = X.column(col).data();
                    result = result.withColumn(col, data);
                }
            }
        }

        return result;
    }

    /** Convenience fit+transform */
    @Override
    public DataFrame fitTransform(DataFrame X) throws Exception {
        fit(X);
        return transform(X);
    }

    public List<String> getTransformerNames() { return Collections.unmodifiableList(names); }
    public BaseTransformer getTransformer(String name) {
        int idx = names.indexOf(name);
        if (idx < 0) throw new IllegalArgumentException("No transformer: " + name);
        return transformers.get(idx);
    }
}

