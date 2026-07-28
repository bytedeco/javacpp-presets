package org.bytedeco.pytorch.dataframe.ml.clustering;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 均值漂移聚类 (Mean Shift)
 * 使用核密度估计的聚类算法
 */
public class MeanShift extends BaseClusterer {
    private double bandwidth = 1.0;
    private int maxIter = 100;
    private double tolerance = 1e-4;
    private double[][] centers;
    private int nClusters = 0;

    public MeanShift(String... features) {
        super(features);
    }

    public MeanShift(double bandwidth, String... features) {
        super(features);
        this.bandwidth = bandwidth;
    }

    @Override
    public MeanShift fit(DataFrame X) {
        double[][] data = extractMatrix(X);
        int n = data.length;

        // 为每个点计算其漂移向量
        double[][] shifted = new double[n][];
        for (int i = 0; i < n; i++) {
            shifted[i] = meanShift(data[i], data);
        }

        // 聚类相似的点
        labels = new int[n];
        Arrays.fill(labels, -1);
        List<double[]> clusterCenters = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (labels[i] == -1) {
                boolean found = false;
                for (int j = 0; j < clusterCenters.size(); j++) {
                    if (euclideanDistance(shifted[i], clusterCenters.get(j)) < bandwidth * 0.5) {
                        labels[i] = j;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    labels[i] = clusterCenters.size();
                    clusterCenters.add(shifted[i]);
                }
            }
        }

        nClusters = clusterCenters.size();
        centers = clusterCenters.toArray(new double[0][]);

        fitted = true;
        return this;
    }

    /**
     * 计算均值漂移向量
     */
    private double[] meanShift(double[] point, double[][] data) {
        double[] shifted = new double[point.length];

        for (int iter = 0; iter < maxIter; iter++) {
            double[] numerator = new double[point.length];
            double denominator = 0;

            for (int j = 0; j < data.length; j++) {
                double dist = euclideanDistance(point, data[j]);
                double weight = Math.exp(-dist * dist / (2 * bandwidth * bandwidth));

                for (int k = 0; k < point.length; k++) {
                    numerator[k] += weight * data[j][k];
                }
                denominator += weight;
            }

            double[] newPoint = new double[point.length];
            for (int k = 0; k < point.length; k++) {
                newPoint[k] = numerator[k] / Math.max(denominator, 1e-10);
            }

            // 检查收敛
            if (euclideanDistance(newPoint, point) < tolerance) {
                shifted = newPoint;
                break;
            }

            point = newPoint;
            shifted = newPoint;
        }

        return shifted;
    }

    @Override
    public int[] predict(DataFrame X) {
        if (!fitted) {
            throw new IllegalStateException("模型未拟合");
        }

        double[][] data = extractMatrix(X);
        int n = data.length;
        int[] predictions = new int[n];

        for (int i = 0; i < n; i++) {
            double minDist = Double.POSITIVE_INFINITY;
            int nearestCluster = 0;

            for (int k = 0; k < nClusters; k++) {
                double dist = euclideanDistance(data[i], centers[k]);
                if (dist < minDist) {
                    minDist = dist;
                    nearestCluster = k;
                }
            }

            predictions[i] = nearestCluster;
        }

        return predictions;
    }

    @Override
    public int getNCluster() {
        return nClusters;
    }

    /**
     * 设置带宽
     */
    public MeanShift setBandwidth(double bandwidth) {
        this.bandwidth = bandwidth;
        return this;
    }
}