package org.bytedeco.pytorch.data.dataframe.ml.clustering;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * 小批量 K-均值 (MiniBatch K-Means)
 * 内存高效的 K-Means 变体，适合大规模数据
 */
public class MiniBatchKMeans extends BaseClusterer {
    private int nClusters = 3;
    private int maxIter = 100;
    private int batchSize = 100;
    private double tolerance = 1e-4;
    private double[][] centers;
    private int[] clusterSizes;

    public MiniBatchKMeans(int nClusters, String... features) {
        super(features);
        this.nClusters = nClusters;
    }

    public MiniBatchKMeans(int nClusters, int batchSize, String... features) {
        super(features);
        this.nClusters = nClusters;
        this.batchSize = batchSize;
    }

    @Override
    public MiniBatchKMeans fit(DataFrame X) {
        double[][] data = extractMatrix(X);
        int n = data.length;

        // 初始化聚类中心
        Random random = new Random(42);
        centers = new double[nClusters][];
        clusterSizes = new int[nClusters];

        Set<Integer> selected = new HashSet<>();
        for (int i = 0; i < nClusters; i++) {
            int idx;
            do {
                idx = random.nextInt(n);
            } while (selected.contains(idx));

            selected.add(idx);
            centers[i] = data[idx].clone();
        }

        // MiniBatch 迭代
        for (int iter = 0; iter < maxIter; iter++) {
            // 随机选择一个批次
            int[] batch = new int[Math.min(batchSize, n)];
            for (int i = 0; i < batch.length; i++) {
                batch[i] = random.nextInt(n);
            }

            // 分配批次中的样本
            int[] assignment = new int[batch.length];
            for (int i = 0; i < batch.length; i++) {
                double minDist = Double.POSITIVE_INFINITY;
                int nearestCluster = 0;

                for (int k = 0; k < nClusters; k++) {
                    double dist = euclideanDistance(data[batch[i]], centers[k]);
                    if (dist < minDist) {
                        minDist = dist;
                        nearestCluster = k;
                    }
                }

                assignment[i] = nearestCluster;
            }

            // 更新聚类中心（小批量更新）
            for (int i = 0; i < batch.length; i++) {
                int cluster = assignment[i];
                clusterSizes[cluster]++;

                double learningRate = 1.0 / clusterSizes[cluster];
                for (int j = 0; j < data[0].length; j++) {
                    centers[cluster][j] += learningRate * (data[batch[i]][j] - centers[cluster][j]);
                }
            }
        }

        // 最终分配
        labels = new int[n];
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

            labels[i] = nearestCluster;
        }

        fitted = true;
        return this;
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
     * 设置批次大小
     */
    public MiniBatchKMeans setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }
}