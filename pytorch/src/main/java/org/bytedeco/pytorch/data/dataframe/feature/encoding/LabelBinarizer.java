package org.bytedeco.pytorch.data.dataframe.feature.encoding;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 标签二值化器 (Label Binarizer)
 * 将多类标签转换为二值矩阵
 * 用于多类分类问题
 */
public class LabelBinarizer extends BaseTransformer {
    private String column;
    private List<Object> classes;
    private Map<Object, Integer> classIndex;

    public LabelBinarizer(String column) {
//        super(columns);

        this.column = column;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        List<Object> values = X.column(column).data();

        // 获取所有唯一的类别
        classes = values.stream()
                .filter(v -> v != null)
                .distinct()
                .sorted((a, b) -> a.toString().compareTo(b.toString()))
                .collect(Collectors.toList());

        // 创建类别索引映射
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

        // 为每个类别创建一列二值特征
        for (int i = 0; i < classes.size(); i++) {
            final Object classLabel = classes.get(i);
            final int classIdx = i;

            List<Integer> binarized = new ArrayList<>();
            for (Object value : X.column(column).data()) {
                binarized.add((value != null && value.equals(classLabel)) ? 1 : 0);
            }

            result = result.withColumn("class_" + classIdx + "_" + classLabel, binarized);
        }

        return result;
    }

    /**
     * 获取类别列表
     */
    public List<Object> getClasses() {
        return new ArrayList<>(classes);
    }

    /**
     * 获取类别个数
     */
    public int getNumClasses() {
        return classes.size();
    }
}