package org.bytedeco.pytorch.data.dataframe.feature.decomposition;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;

/**
 * 截断奇异值分解 (Truncated SVD)
 * 用于降维，特别是在稀疏数据上有效
 */
public class TruncatedSVD extends BaseTransformer {
    private int nComponents;
    private double[][] components;
    private double[] singularValues;
    private double[] mean;
    private String[] columns;

    public TruncatedSVD(int nComponents, String... columns) {
        super(columns);
        if (nComponents <= 0) {
            throw new IllegalArgumentException("nComponents 必须大于 0");
        }
        this.nComponents = nComponents;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        // 提取数值矩阵
        double[][] matrix = extractMatrix(X);

        // 计算均值并中心化
        mean = computeMean(matrix);
        centerMatrix(matrix, mean);

        // 进行 SVD 分解
        SVDResult svdResult = performSVD(matrix, nComponents);
        this.components = svdResult.vectors;
        this.singularValues = svdResult.singularValues;

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

        // 投影到 SVD 分量上
        // components shape: [nFeatures][nComponents]
        // 循环: k 是分量索引，col 是特征索引
        for (int k = 0; k < nComponents; k++) {
            List<Double> projected = new ArrayList<>();

            for (int row = 0; row < matrix.length; row++) {
                double projection = 0;
                for (int col = 0; col < columns.length; col++) {
                    // ✅ 正确：col 在前（特征），k 在后（分量）
                    projection += matrix[row][col] * components[col][k];
                }
                projected.add(projection);
            }

            result = result.withColumn("SVD" + (k + 1), projected);
        }

        return result;
    }

    /**
     * 获取奇异值
     */
    public double[] getSingularValues() {
        return singularValues;
    }

    /**
     * 获取解释方差比例
     */
    public double[] getExplainedVarianceRatio() {
        if (singularValues == null) {
            throw new IllegalStateException("转换器未拟合");
        }

        double[] ratios = new double[singularValues.length];
        double totalVariance = 0;

        for (double sv : singularValues) {
            totalVariance += sv * sv;
        }

        for (int i = 0; i < singularValues.length; i++) {
            ratios[i] = (singularValues[i] * singularValues[i]) / totalVariance;
        }

        return ratios;
    }

    private double[][] extractMatrix(DataFrame X) {
        double[][] matrix = new double[X.rowCount()][columns.length];
        for (int i = 0; i < columns.length; i++) {
            List<Object> col = X.column(columns[i]).data();
            for (int j = 0; j < col.size(); j++) {
                matrix[j][i] = DataValues.asDouble(col.get(j));
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

    /**
     * 简化的 SVD 实现 (使用幂迭代法近似)
     */
    private SVDResult performSVD(double[][] matrix, int nComponents) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        nComponents = Math.min(nComponents, Math.min(rows, cols));

        double[][] U = new double[rows][nComponents];
        double[] singularValues = new double[nComponents];
        double[][] V = new double[cols][nComponents];

        // 计算 AAT 矩阵用于提取左奇异向量
        double[][] aat = multiplyMatrices(matrix, transposeMatrix(matrix));

        // 使用幂迭代法计算特征向量
        for (int k = 0; k < nComponents; k++) {
            double[] v = powerIteration(aat, 100);

            // 归一化
            double norm = vectorNorm(v);
            for (int i = 0; i < v.length; i++) {
                U[i][k] = v[i] / norm;
            }

            singularValues[k] = Math.sqrt(norm);

            // 计算对应的右奇异向量
            double[] rightVector = new double[cols];
            for (int i = 0; i < cols; i++) {
                double sum = 0;
                for (int j = 0; j < rows; j++) {
                    sum += matrix[j][i] * U[j][k];
                }
                rightVector[i] = sum / Math.max(singularValues[k], 1e-10);
            }
            for (int i = 0; i < cols; i++) {
                V[i][k] = rightVector[i];
            }

            // 放大矩阵（Deflation）
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix[i][j] -= singularValues[k] * U[i][k] * V[j][k];
                }
            }
        }

        return new SVDResult(V, singularValues);
    }

    private double[][] multiplyMatrices(double[][] A, double[][] B) {
        int rows = A.length;
        int cols = B[0].length;
        double[][] C = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double sum = 0;
                for (int k = 0; k < A[0].length; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    }

    private double[][] transposeMatrix(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    private double[] powerIteration(double[][] matrix, int iterations) {
        int n = matrix.length;
        double[] v = new double[n];

        // 初始化随机向量
        for (int i = 0; i < n; i++) {
            v[i] = Math.random();
        }

        for (int iter = 0; iter < iterations; iter++) {
            double[] Av = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    Av[i] += matrix[i][j] * v[j];
                }
            }

            double norm = vectorNorm(Av);
            if (norm > 1e-10) {
                for (int i = 0; i < n; i++) {
                    v[i] = Av[i] / norm;
                }
            }
        }

        return v;
    }

    private double vectorNorm(double[] v) {
        double sum = 0;
        for (double val : v) {
            sum += val * val;
        }
        return Math.sqrt(sum);
    }

    private static class SVDResult {
        double[][] vectors;
        double[] singularValues;

        SVDResult(double[][] vectors, double[] singularValues) {
            this.vectors = vectors;
            this.singularValues = singularValues;
        }
    }
}

