package org.bytedeco.pytorch.data.dataframe.feature.text;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * 哈希向量化器 (Hashing Vectorizer)
 * 使用哈希技巧将文本转换为固定维度的向量
 * 内存效率高，适合大规模文本数据
 */
public class HashingVectorizer extends BaseTransformer {
    private String column;
    private int nFeatures = 128;
    private String pattern = "\\b\\w+\\b";
    private boolean lowerCase = true;

    public HashingVectorizer(String column) {
//        super(columns);
        this.column = column;
    }

    public HashingVectorizer(String column, int nFeatures) {
        this.column = column;
        this.nFeatures = nFeatures;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        // 哈希向量化不需要拟合参数
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        DataFrame result = X.copy();
        List<Object> texts = X.column(column).data();

        // 创建特征列
        for (int i = 0; i < nFeatures; i++) {
            final int featureIdx = i;
            List<Double> featureValues = new ArrayList<>();

            for (Object text : texts) {
                String doc = text.toString();
                if (lowerCase) {
                    doc = doc.toLowerCase();
                }

                double value = hashText(doc, featureIdx);
                featureValues.add(value);
            }

            result = result.withColumn("hash_" + i, featureValues);
        }

        return result;
    }

    /**
     * 哈希文本到特定特征
     */
    private double hashText(String text, int featureIdx) {
        Set<String> tokens = tokenize(text);
        double value = 0;

        for (String token : tokens) {
            int hash = Math.abs(token.hashCode()) % nFeatures;
            if (hash == featureIdx) {
                value += 1.0 / tokens.size();
            }
        }

        return value;
    }

    /**
     * 分词
     */
    private Set<String> tokenize(String text) {
        Set<String> tokens = new HashSet<>();
        Pattern p = Pattern.compile(pattern);
        java.util.regex.Matcher m = p.matcher(text);

        while (m.find()) {
            String token = m.group();
            if (token.length() > 0) {
                tokens.add(token);
            }
        }

        return tokens;
    }

    /**
     * 设置特征数
     */
    public void setNFeatures(int nFeatures) {
        this.nFeatures = nFeatures;
    }
}