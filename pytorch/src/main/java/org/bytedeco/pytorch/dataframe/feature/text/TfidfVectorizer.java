package org.bytedeco.pytorch.dataframe.feature.text;

 import org.bytedeco.pytorch.dataframe.DataFrame;
  import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * TF-IDF 向量化器 (TF-IDF Vectorizer)
 * 将文本转换为 TF-IDF 特征向量
 * 用于自然语言处理和文本分类任务
 */
public class TfidfVectorizer extends BaseTransformer {
    private String column;
    private int maxFeatures;
    private int minDf;
    private int maxDf;
    private boolean lowerCase;
    private String pattern;

    private Map<String, Integer> vocabulary = new HashMap<>();
    private Map<String, Double> idfValues = new HashMap<>();
    private List<String> featureNames;

    /**
     * @param column 要处理的文本列
     * @param maxFeatures 最多保留的特征数
     * @param minDf 最少出现次数（绝对值）
     * @param maxDf 最多出现次数（绝对值）
     */
    public TfidfVectorizer(String column, int maxFeatures, int minDf, int maxDf) {
        this.column = column;
        this.maxFeatures = maxFeatures;
        this.minDf = minDf;
        this.maxDf = maxDf;
        this.lowerCase = true;
        this.pattern = "\\b\\w+\\b"; // 单词分词模式
    }

    public TfidfVectorizer(String column) {
//        super(columns);
        this(column, 1000, 1, Integer.MAX_VALUE);
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        List<Object> texts = X.column(column).data();
        int docCount = texts.size();

        // 统计词频
        Map<String, Integer> wordDocFreq = new HashMap<>();
        Map<String, Integer> wordFreq = new HashMap<>();

        for (Object text : texts) {
            String doc = text.toString();
            if (lowerCase) {
                doc = doc.toLowerCase();
            }

            // 分词
            Set<String> docWords = tokenize(doc);

            for (String word : docWords) {
                // 文档频率（有多少个文档包含这个词）
                wordDocFreq.put(word, wordDocFreq.getOrDefault(word, 0) + 1);
            }

            // 词频（总共出现多少次）
            for (String word : docWords) {
                wordFreq.put(word, wordFreq.getOrDefault(word, 0) + 1);
            }
        }

        // 过滤词汇（根据 minDf 和 maxDf）
        Map<String, Integer> filteredWords = wordDocFreq.entrySet().stream()
                .filter(e -> e.getValue() >= minDf && e.getValue() <= maxDf)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        // 按频率排序，保留 maxFeatures 个
        featureNames = filteredWords.entrySet().stream()
                .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                .limit(maxFeatures)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());

        // 构建词汇表
        for (int i = 0; i < featureNames.size(); i++) {
            vocabulary.put(featureNames.get(i), i);
        }

        // 计算 IDF 值 (log(N / df) + 1)
        for (String word : featureNames) {
            int df = wordDocFreq.get(word);
            double idf = Math.log((double) docCount / df) + 1.0;
            idfValues.put(word, idf);
        }

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        List<Object> texts = X.column(column).data();
        DataFrame result = X.copy();

        // 为每个特征创建一列
        for (String word : featureNames) {
            List<Double> tfidfScores = new ArrayList<>();

            for (Object text : texts) {
                String doc = text.toString();
                if (lowerCase) {
                    doc = doc.toLowerCase();
                }

                double tf = calculateTF(doc, word);
                double idf = idfValues.get(word);
                double tfidfScore = tf * idf;

                tfidfScores.add(tfidfScore);
            }

            result = result.withColumn("tfidf_" + word, tfidfScores);
        }

        return result;
    }

    /**
     * 计算 TF 值
     */
    private double calculateTF(String doc, String word) {
        Set<String> words = tokenize(doc);
        int count = 0;

        for (String w : words) {
            if (w.equals(word)) {
                count++;
            }
        }

        // TF 是词在文档中出现次数 / 文档总词数
        return (double) count / words.size();
    }

    /**
     * 文本分词
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
     * 获取特征名称
     */
    public List<String> getFeatureNames() {
        return new ArrayList<>(featureNames);
    }

    /**
     * 获取���汇表
     */
    public Map<String, Integer> getVocabulary() {
        return new HashMap<>(vocabulary);
    }

    /**
     * 获取 IDF 值
     */
    public Map<String, Double> getIdfValues() {
        return new HashMap<>(idfValues);
    }

    /**
     * 获取特征数
     */
    public int getFeatureCount() {
        return featureNames.size();
    }
}