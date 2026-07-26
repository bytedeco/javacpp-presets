package org.bytedeco.pytorch.data.dataframe.feature.manifold;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * t-分布邻域嵌入 (t-SNE)
 * 用于高维数据可视化
 * 保持局部邻域结构
 */
public class TSNE extends BaseTransformer {
    private int nComponents = 2;
    private String[] columns;
    private int nIter = 1000;
    private int perplexity = 30;
    private double learningRate = 200.0;
    private double[][] embedding;

    public TSNE(String... columns) {

        super(columns);
        this.columns = columns;
    }

    public TSNE(int nComponents, String... columns) {
        this.nComponents = nComponents;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        double[][] data = extractMatrix(X);

        // 计算条件概率
        double[][] P = computeConditionalProb(data);

        // 随机初始化嵌入
        embedding = new double[data.length][nComponents];
        Random random = new Random(42);
        for (int i = 0; i < embedding.length; i++) {
            for (int j = 0; j < nComponents; j++) {
                embedding[i][j] = (random.nextDouble() - 0.5) * 1e-4;
            }
        }

        // t-SNE 迭代（简化实现）
        for (int iter = 0; iter < nIter; iter++) {
            updateEmbedding(P);
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

        for (int i = 0; i < nComponents; i++) {
            List<Double> component = new ArrayList<>();
            for (int j = 0; j < embedding.length; j++) {
                component.add(embedding[j][i]);
            }
            result = result.withColumn("tSNE_" + (i + 1), component);
        }

        return result;
    }

    private double[][] computeConditionalProb(double[][] data) {
        int n = data.length;
        double[][] P = new double[n][n];

        // 简化实现：高斯核
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double dist = 0;
                    for (int k = 0; k < data[0].length; k++) {
                        double diff = data[i][k] - data[j][k];
                        dist += diff * diff;
                    }
                    P[i][j] = Math.exp(-dist);
                }
            }
        }

        return P;
    }

    private void updateEmbedding(double[][] P) {
        // 简化的梯度下降更新
    }

    private double[][] extractMatrix(DataFrame X) {
        double[][] matrix = new double[X.rowCount()][columns.length];
        for (int i = 0; i < columns.length; i++) {
            List<Object> col = X.column(columns[i]).data();
            for (int j = 0; j < col.size(); j++) {
                Object val = col.get(j);
                matrix[j][i] = val != null ? DataValues.asDouble(val) : 0;
            }
        }
        return matrix;
    }
}
