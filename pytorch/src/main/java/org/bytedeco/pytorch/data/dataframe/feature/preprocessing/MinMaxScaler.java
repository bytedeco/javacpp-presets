package org.bytedeco.pytorch.data.dataframe.feature.preprocessing;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
 import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class MinMaxScaler extends BaseTransformer {
    private Map<String, Double> mins = new HashMap<>();
    private Map<String, Double> maxs = new HashMap<>();
    private String[] columns;
    private double minValue = 0.0;
    private double maxValue = 1.0;

    public MinMaxScaler(String... columns) {
        super(columns);
        this.columns = columns;
    }

    public MinMaxScaler setRange(double min, double max) {
        this.minValue = min;
        this.maxValue = max;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (String col : columns) {
            List<Double> values = X.column(col).data().stream()
                .map(v ->  DataValues.asDouble(v))
                .collect(Collectors.toList());

            double min = values.stream().mapToDouble(v -> v).min().orElse(0.0);
            double max = values.stream().mapToDouble(v -> v).max().orElse(1.0);

            mins.put(col, min);
            maxs.put(col, max);
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("未拟合");

        DataFrame result = X.copy();
        for (String col : columns) {
            double min = mins.get(col);
            double max = maxs.get(col);
            double range = max - min;

            result = result.withColumn(col + "_minmax",
                X.column(col).data().stream()
                    .map(v -> {
                        double normalized = (DataValues.asDouble(v) - min) / range;
                        return minValue + normalized * (maxValue - minValue);
                    })
                    .collect(Collectors.toList()));
        }
        return result;
    }
}