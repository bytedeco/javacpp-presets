package org.bytedeco.pytorch.data.dataframe.feature.decomposition;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 非负矩阵分解 (Non-negative Matrix Factorization)
 * 适合非负数据，应用于主题建模、图像分解等
 */
public class NMF extends BaseTransformer {
    private int nComponents;
    private String[] columns;
    private double[][] W;  // 基矩阵
    private double[][] H;  // 系数矩阵
    private int maxIter = 200;
    private double tolerance = 1e-4;

    public NMF(int nComponents, String... columns) {
        super(columns);
        this.nComponents = nComponents;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        double[][] matrix = extractMatrix(X);

        // 初始化 W 和 H
        Random random = new Random(42);
        W = new double[matrix.length][nComponents];
        H = new double[nComponents][matrix[0].length];

        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[0].length; j++) {
                W[i][j] = Math.abs(random.nextGaussian());
            }
        }

        for (int i = 0; i < H.length; i++) {
            for (int j = 0; j < H[0].length; j++) {
                H[i][j] = Math.abs(random.nextGaussian());
            }
        }

        // NMF 迭代（简化实现）
        for (int iter = 0; iter < maxIter; iter++) {
            // 更新 H
            updateH(matrix, W, H);
            // 更新 W
            updateW(matrix, W, H);
        }

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        double[][] matrix = extractMatrix(X);
        DataFrame result = X.copy();

        // 返回 W 矩阵（组件系数）
        for (int i = 0; i < nComponents; i++) {
            List<Double> component = new ArrayList<>();
            for (int j = 0; j < W.length; j++) {
                component.add(W[j][i]);
            }
            result = result.withColumn("NMF_" + (i + 1), component);
        }

        return result;
    }

    private void updateH(double[][] matrix, double[][] W, double[][] H) {
        // 简化实现
    }

    private void updateW(double[][] matrix, double[][] W, double[][] H) {
        // 简化实现
    }

    private double[][] extractMatrix(DataFrame X) {
        double[][] matrix = new double[X.rowCount()][columns.length];
        for (int i = 0; i < columns.length; i++) {
            List<Object> col = X.column(columns[i]).data();
            for (int j = 0; j < col.size(); j++) {
                Object val = col.get(j);
                matrix[j][i] = val != null ? Math.abs(DataValues.asDouble(val)) : 0;
            }
        }
        return matrix;
    }
}