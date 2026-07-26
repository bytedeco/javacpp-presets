package org.bytedeco.pytorch.data.dataframe.ml.clustering;

import org.bytedeco.pytorch.data.dataframe.DataValues;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.util.List;

/**
 * 聚类算法基类
 * 所有聚类算法的父类
 */
public abstract class BaseClusterer {
    protected String[] features;
    protected int[] labels;
    protected boolean fitted = false;

    public BaseClusterer(String... features) {
        this.features = features;
    }

    /**
     * 拟合聚类模型
     */
    public abstract BaseClusterer fit(DataFrame X);

    /**
     * 预测新数据的标签
     */
    public abstract int[] predict(DataFrame X);

    /**
     * 拟合并预测
     */
    public int[] fitPredict(DataFrame X) {
        fit(X);
        return predict(X);
    }

    /**
     * 获取聚类标签
     */
    public int[] getLabels() {
        if (!fitted) {
            throw new IllegalStateException("模型未拟合");
        }
        return labels;
    }

    /**
     * 获取聚类数
     */
    public abstract int getNCluster();

    /**
     * 检查是否拟合
     */
    public boolean isFitted() {
        return fitted;
    }

    /**
     * 提取特征矩阵
     */
    protected double[][] extractMatrix(DataFrame X) {
        double[][] matrix = new double[X.rowCount()][features.length];
        for (int i = 0; i < features.length; i++) {
            List<Object> col = X.column(features[i]).data();
            for (int j = 0; j < col.size(); j++) {
                double d = DataValues.asDouble(col.get(j));
                matrix[j][i] = Double.isNaN(d) ? 0 : d;
            }
        }
        return matrix;
    }

    /**
     * 计算欧氏距离
     */
    protected double euclideanDistance(double[] p1, double[] p2) {
        double sum = 0;
        for (int i = 0; i < p1.length; i++) {
            double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * 计算曼哈顿距离
     */
    protected double manhattanDistance(double[] p1, double[] p2) {
        double sum = 0;
        for (int i = 0; i < p1.length; i++) {
            sum += Math.abs(p1[i] - p2[i]);
        }
        return sum;
    }

    /**
     * 计算距离矩阵
     */
    protected double[][] computeDistanceMatrix(double[][] data) {
        int n = data.length;
        double[][] distances = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    distances[i][j] = 0;
                } else {
                    double dist = euclideanDistance(data[i], data[j]);
                    distances[i][j] = dist;
                    distances[j][i] = dist;
                }
            }
        }
        return distances;
    }
}