package org.bytedeco.pytorch.dataframe.feature.encoding;

 import org.bytedeco.pytorch.dataframe.DataFrame;
 import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class LabelEncoder extends BaseTransformer {
    private Map<String, Map<Object, Integer>> labelMaps = new HashMap<>();
    private String[] columns;

    public LabelEncoder(String... columns) {
        super(columns);
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (String col : columns) {
            Set<Object> uniqueValues = new HashSet<>(X.column(col).data());
            Map<Object, Integer> labelMap = new HashMap<>();

            int idx = 0;
            for (Object value : uniqueValues) {
                labelMap.put(value, idx++);
            }

            labelMaps.put(col, labelMap);
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("未拟合");

        DataFrame result = X.copy();

        for (String col : columns) {
            Map<Object, Integer> labelMap = labelMaps.get(col);

            result = result.withColumn(col + "_encoded",
                X.column(col).data().stream()
                    .map(v -> labelMap.getOrDefault(v, -1))
                    .collect(Collectors.toList()));
        }
        return result;
    }
}
