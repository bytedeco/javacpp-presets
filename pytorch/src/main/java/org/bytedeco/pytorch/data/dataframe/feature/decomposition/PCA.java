package org.bytedeco.pytorch.data.dataframe.feature.decomposition;

import org.bytedeco.pytorch.data.dataframe.DataValues;


 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;

public class PCA extends BaseTransformer {
    private int nComponents;
    private double[][] components;
    private double[] mean;
    private String[] columns;

    public PCA(int nComponents, String... columns) {
        super(columns);
        this.nComponents = nComponents;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        // 提取数值矩阵
        double[][] matrix = new double[X.rowCount()][columns.length];
        for (int i = 0; i < columns.length; i++) {
            List<Object> col = X.column(columns[i]).data();
            for (int j = 0; j < col.size(); j++) {
                matrix[j][i] = DataValues.asDouble(col.get(j));
            }
        }

        // 计算均值
        mean = new double[columns.length];
        for (int i = 0; i < columns.length; i++) {
            double sum = 0;
            for (int j = 0; j < matrix.length; j++) {
                sum += matrix[j][i];
            }
            mean[i] = sum / matrix.length;
        }

        // 中心化
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < columns.length; j++) {
                matrix[i][j] -= mean[j];
            }
        }

        // 计算协方差矩阵
        double[][] cov = computeCovariance(matrix);

        // 特征值分解 (简化实现)
        components = computeEigenVectors(cov, nComponents);

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("未拟合");

        DataFrame result = X.copy();

        // 项目数据到主成分
        for (int pc = 0; pc < nComponents; pc++) {
            List<Double> projected = new ArrayList<>();

            for (int row = 0; row < X.rowCount(); row++) {
                double projection = 0;
                for (int col = 0; col < columns.length; col++) {
                    double value = DataValues.asDouble(X.column(columns[col]).get(row));
                    projection += (value - mean[col]) * components[pc][col];
                }
                projected.add(projection);
            }

            result = result.withColumn("PC" + (pc + 1), projected);
        }

        return result;
    }

    private double[][] computeCovariance(double[][] matrix) {
        int cols = matrix[0].length;
        double[][] cov = new double[cols][cols];

        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                double sum = 0;
                for (int k = 0; k < matrix.length; k++) {
                    sum += matrix[k][i] * matrix[k][j];
                }
                cov[i][j] = sum / (matrix.length - 1);
            }
        }
        return cov;
    }

    private double[][] computeEigenVectors(double[][] matrix, int nComponents) {
        // 简化实现：返回单位矩阵的前 n 行
        double[][] vectors = new double[nComponents][matrix.length];
        for (int i = 0; i < nComponents; i++) {
            for (int j = 0; j < matrix.length; j++) {
                vectors[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
        return vectors;
    }
}
