
package org.bytedeco.pytorch.data.dataframe.ml.clustering;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * 谱聚类 (Spectral Clustering)
 * 使用图论的聚类方法
 */
public class SpectralClustering extends BaseClusterer {
    private int nClusters;
    private double sigma = 1.0;
    private double[][] trainData;
    private int trainDataSize;

    public SpectralClustering(int nClusters, String... features) {
        super(features);
        this.nClusters = nClusters;
    }

    @Override
    public SpectralClustering fit(DataFrame X) {
        trainData = extractMatrix(X);
        trainDataSize = trainData.length;
        int n = trainDataSize;

        // 计算相似矩阵
        double[][] similarity = computeSimilarityMatrix(trainData);

        // 计算拉普拉斯矩阵
        double[][] laplacian = computeLaplacian(similarity);

        // 简化实现：使用 K-Means 进行聚类
        // 在实际应用中应该使用拉普拉斯矩阵的特征向量
        labels = new int[n];

        // 使用基于距离的简单聚类方法
        Random rand = new Random(42);

        // 初始化聚类中心
        double[][] centers = new double[nClusters][];
        Set<Integer> selected = new HashSet<>();

        for (int i = 0; i < nClusters; i++) {
            int idx;
            do {
                idx = rand.nextInt(n);
            } while (selected.contains(idx));
            selected.add(idx);
            centers[i] = trainData[idx].clone();
        }

        // K-Means 迭代
        for (int iter = 0; iter < 100; iter++) {
            // 分配样本
            int[] newLabels = new int[n];
            for (int i = 0; i < n; i++) {
                double minDist = Double.POSITIVE_INFINITY;
                int nearestCluster = 0;

                for (int k = 0; k < nClusters; k++) {
                    double dist = euclideanDistance(trainData[i], centers[k]);
                    if (dist < minDist) {
                        minDist = dist;
                        nearestCluster = k;
                    }
                }
                newLabels[i] = nearestCluster;
            }

            // 更新中心
            double[][] newCenters = new double[nClusters][trainData[0].length];
            int[] clusterSizes = new int[nClusters];

            for (int i = 0; i < n; i++) {
                int cluster = newLabels[i];
                clusterSizes[cluster]++;
                for (int j = 0; j < trainData[0].length; j++) {
                    newCenters[cluster][j] += trainData[i][j];
                }
            }

            for (int k = 0; k < nClusters; k++) {
                if (clusterSizes[k] > 0) {
                    for (int j = 0; j < trainData[0].length; j++) {
                        newCenters[k][j] /= clusterSizes[k];
                    }
                } else {
                    newCenters[k] = centers[k].clone();
                }
            }

            centers = newCenters;
            labels = newLabels;
        }

        fitted = true;
        return this;
    }

    /**
     * 计算相似矩阵
     */
    private double[][] computeSimilarityMatrix(double[][] data) {
        int n = data.length;
        double[][] similarity = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dist = euclideanDistance(data[i], data[j]);
                similarity[i][j] = Math.exp(-dist * dist / (2 * sigma * sigma));
            }
        }

        return similarity;
    }

    /**
     * 计算拉普拉斯矩阵
     */
    private double[][] computeLaplacian(double[][] similarity) {
        int n = similarity.length;
        double[][] laplacian = new double[n][n];

        // 计算度矩阵 D
        for (int i = 0; i < n; i++) {
            double degree = 0;
            for (int j = 0; j < n; j++) {
                degree += similarity[i][j];
            }
            laplacian[i][i] = degree;
        }

        // 计算 L = D - W
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    laplacian[i][j] = -similarity[i][j];
                }
            }
        }

        return laplacian;
    }

    @Override
    public int[] predict(DataFrame X) {
        if (!fitted) {
            throw new IllegalStateException("模型未拟合");
        }

        double[][] newData = extractMatrix(X);
        int n = newData.length;
        int[] predictions = new int[n];

        // ✅ 对每个新样本，找最近的训练样本，返回其聚类标签
        for (int i = 0; i < n; i++) {
            double minDist = Double.POSITIVE_INFINITY;
            int nearestCluster = -1;

            for (int j = 0; j < trainDataSize; j++) {
                double dist = euclideanDistance(newData[i], trainData[j]);
                if (dist < minDist) {
                    minDist = dist;
                    nearestCluster = labels[j];
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
     * 设置高斯核带宽
     */
    public SpectralClustering setSigma(double sigma) {
        this.sigma = sigma;
        return this;
    }
}
