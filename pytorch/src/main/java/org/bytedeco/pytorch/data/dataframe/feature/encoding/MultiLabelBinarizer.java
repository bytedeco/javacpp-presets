package org.bytedeco.pytorch.data.dataframe.feature.encoding;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 多标签二值化器 (Multi Label Binarizer)
 * 处理多标签分类问题，每个样本可以有多个标签
 */
public class MultiLabelBinarizer extends BaseTransformer {
    private String column;
    private List<Object> classes;
    private Map<Object, Integer> classIndex;

    public MultiLabelBinarizer(String column) {
//        super(columns);
        this.column = column;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        List<Object> values = X.column(column).data();
        Set<Object> allClasses = new HashSet<>();

        // 提取所有标签
        for (Object value : values) {
            if (value != null && value instanceof Collection) {
                allClasses.addAll((Collection<?>) value);
            }
        }

        // 排序类别
        classes = allClasses.stream()
                .sorted((a, b) -> a.toString().compareTo(b.toString()))
                .collect(Collectors.toList());

        // 创建索引映射
        classIndex = new HashMap<>();
        for (int i = 0; i < classes.size(); i++) {
            classIndex.put(classes.get(i), i);
        }

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        DataFrame result = X.copy();

        // 为每个标签创建一列
        for (int i = 0; i < classes.size(); i++) {
            final Object classLabel = classes.get(i);

            List<Integer> binarized = new ArrayList<>();
            for (Object value : X.column(column).data()) {
                if (value != null && value instanceof Collection) {
                    Collection<?> labels = (Collection<?>) value;
                    binarized.add(labels.contains(classLabel) ? 1 : 0);
                } else {
                    binarized.add(0);
                }
            }

            result = result.withColumn("label_" + i + "_" + classLabel, binarized);
        }

        return result;
    }

    /**
     * 获取类别列表
     */
    public List<Object> getClasses() {
        return new ArrayList<>(classes);
    }
}