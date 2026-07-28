package org.bytedeco.pytorch.dataframe.feature.selection;

import org.bytedeco.pytorch.dataframe.DataValues;

 import org.bytedeco.pytorch.dataframe.DataFrame;
 import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;


public class VarianceThreshold extends BaseTransformer {
    private double threshold;
    private List<String> selectedColumns = new ArrayList<>();
    private String[] columns;

    public VarianceThreshold(double threshold, String... columns) {
        super(columns);
        this.threshold = threshold;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (String col : columns) {
            List<Double> values = X.column(col).data().stream()
                .map(v -> DataValues.asDouble(v))
                .collect(Collectors.toList());

            double mean = values.stream().mapToDouble(v -> v).average().orElse(0.0);
            double variance = values.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .average().orElse(0.0);

            if (variance > threshold) {
                selectedColumns.add(col);
            }
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("未拟合");
        return X.select(selectedColumns.toArray(new String[0]));
    }
}
