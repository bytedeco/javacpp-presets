package org.bytedeco.pytorch.data.dataframe.feature.encoding;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.*;
import java.util.stream.Collectors;

public class OrdinalEncoder extends BaseTransformer {
    private Map<String, List<Object>> categories = new HashMap<>();
    private String[] columns;
    private Map<String, Object> handleUnknown = new HashMap<>();
    private Object unknownValue = -1;

    public OrdinalEncoder(String... columns) {

        super(columns);
        this.columns = columns;
    }

    public OrdinalEncoder setUnknownValue(Object unknownValue) {
        this.unknownValue = unknownValue;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (String col : columns) {
            List<Object> columnData = X.column(col).data();
            Set<Object> uniqueValues = new LinkedHashSet<>(columnData);
            categories.put(col, new ArrayList<>(uniqueValues));
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) {
        if (!fitted) throw new IllegalStateException("未拟合");

        try {
            DataFrame result = X.copy();

            for (String col : columns) {
                List<Object> categoryList = categories.get(col);
                List<Object> columnData = X.column(col).data();

                List<Object> encoded = columnData.stream()
                    .map(value -> {
                        if (value == null) return unknownValue;
                        int index = categoryList.indexOf(value);
                        return index >= 0 ? index : unknownValue;
                    })
                    .collect(Collectors.toList());

                result = result.withColumn(col + "_ordinal", encoded);
            }

            return result;
        } catch (Exception e) {
            throw new RuntimeException("OrdinalEncoder transform 失败", e);
        }
    }

    public Map<String, List<Object>> getCategories() {
        return categories;
    }
}