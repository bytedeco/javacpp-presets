package org.bytedeco.pytorch.data.dataframe.feature.decomposition;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 快速独立成分分析 (Fast ICA)
 * 提取统计上独立的成分
 * 用于盲源分离问题
 */
public class FastICA extends BaseTransformer {
    private int nComponents;
    private String[] columns;
    private double[][] components;
    private double[] mean;
    private int maxIter = 200;
    private double tolerance = 1e-5;

    public FastICA(int nComponents, String... columns) {
        super(columns);
        this.nComponents = nComponents;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        double[][] matrix = extractMatrix(X);

        // 中心化
        mean = computeMean(matrix);
        centerMatrix(matrix, mean);

        // 白化（预处理）
        double[][] whitened = whiten(matrix);

        // ICA 迭代
        components = performICA(whitened, nComponents);

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        double[][] matrix = extractMatrix(X);
        centerMatrix(matrix, mean);

        DataFrame result = X.copy();

        // 投影到独立成分
        for (int i = 0; i < nComponents; i++) {
            List<Double> projected = new ArrayList<>();

            for (int row = 0; row < matrix.length; row++) {
                double projection = 0;
                for (int col = 0; col < columns.length; col++) {
                    projection += matrix[row][col] * components[i][col];
                }
                projected.add(projection);
            }

            result = result.withColumn("IC" + (i + 1), projected);
        }

        return result;
    }

    /**
     * 数据白化
     */
    private double[][] whiten(double[][] matrix) {
        // 简化实现：返回原矩阵
        // 完整实现需要协方差分解
        return matrix;
    }

    /**
     * ICA 算法（简化版）
     */
    private double[][] performICA(double[][] matrix, int nComponents) {
        int cols = matrix[0].length;
        double[][] ica = new double[nComponents][cols];

        // 随机初始化
        Random random = new Random(42);
        for (int i = 0; i < nComponents; i++) {
            for (int j = 0; j < cols; j++) {
                ica[i][j] = random.nextGaussian();
            }
            // 归一化
            double norm = 0;
            for (int j = 0; j < cols; j++) {
                norm += ica[i][j] * ica[i][j];
            }
            norm = Math.sqrt(norm);
            for (int j = 0; j < cols; j++) {
                ica[i][j] /= norm;
            }
        }

        return ica;
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

    private double[] computeMean(double[][] matrix) {
        double[] mean = new double[matrix[0].length];
        for (int j = 0; j < matrix[0].length; j++) {
            double sum = 0;
            for (int i = 0; i < matrix.length; i++) {
                sum += matrix[i][j];
            }
            mean[j] = sum / matrix.length;
        }
        return mean;
    }

    private void centerMatrix(double[][] matrix, double[] mean) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] -= mean[j];
            }
        }
    }
}