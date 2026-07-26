package org.bytedeco.pytorch.data.dataframe.ml.clustering;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * K-均值聚类 (K-Means)
 * 基于质心的分割聚类算法
 */
public class KMeans extends BaseClusterer {
    private int nClusters = 3;
    private int maxIter = 300;
    private double tolerance = 1e-4;
    private int nInit = 10;
    private String initMethod = "k-means++";  // random 或 k-means++
    private double[][] centers;
    private double[] inertia;
    private int nIter = 0;

    public KMeans(int nClusters, String... features) {
        super(features);
        this.nClusters = nClusters;
    }

    public KMeans(int nClusters, int maxIter, String... features) {
        super(features);
        this.nClusters = nClusters;
        this.maxIter = maxIter;
    }

    @Override
    public KMeans fit(DataFrame X) {
        double[][] data = extractMatrix(X);
        int n = data.length;

        double bestInertia = Double.POSITIVE_INFINITY;
        double[][] bestCenters = null;
        int[] bestLabels = null;

        // 运行多次初始化
        for (int init = 0; init < nInit; init++) {
            // 初始化聚类中心
            centers = initializeCenters(data, nClusters);

            double prevInertia = Double.POSITIVE_INFINITY;

            // K-means 迭代
            for (int iter = 0; iter < maxIter; iter++) {
                nIter = iter;

                // 分配样本到最近的聚类中心
                int[] assignment = new int[n];
                double currentInertia = 0;

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

                    assignment[i] = nearestCluster;
                    currentInertia += minDist * minDist;
                }

                // 检查收敛
                if (Math.abs(prevInertia - currentInertia) < tolerance) {
                    break;
                }

                // 更新聚类中心
                double[][] newCenters = new double[nClusters][data[0].length];
                int[] clusterSizes = new int[nClusters];

                for (int i = 0; i < n; i++) {
                    int cluster = assignment[i];
                    clusterSizes[cluster]++;
                    for (int j = 0; j < data[0].length; j++) {
                        newCenters[cluster][j] += data[i][j];
                    }
                }

                for (int k = 0; k < nClusters; k++) {
                    if (clusterSizes[k] > 0) {
                        for (int j = 0; j < data[0].length; j++) {
                            newCenters[k][j] /= clusterSizes[k];
                        }
                    }
                }

                centers = newCenters;
                prevInertia = currentInertia;
            }

            // 最后一次分配
            int[] assignment = new int[n];
            double currentInertia = 0;

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

                assignment[i] = nearestCluster;
                currentInertia += minDist * minDist;
            }

            // 保留最好的结果
            if (currentInertia < bestInertia) {
                bestInertia = currentInertia;
                bestLabels = assignment;
                bestCenters = centers.clone();
            }
        }

        labels = bestLabels;
        centers = bestCenters;
        inertia = new double[]{bestInertia};

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

    /**
     * 初始化聚类中心
     */
    private double[][] initializeCenters(double[][] data, int k) {
        if ("k-means++".equals(initMethod)) {
            return initKMeansPlusPlus(data, k);
        } else {
            return initRandom(data, k);
        }
    }

    /**
     * K-means++ 初始化
     */
    private double[][] initKMeansPlusPlus(double[][] data, int k) {
        int n = data.length;
        double[][] centers = new double[k][];
        Random random = new Random(42);

        // 选择第一个中心
        centers[0] = data[random.nextInt(n)].clone();

        // 选择剩余的 k-1 个中心
        for (int i = 1; i < k; i++) {
            double[] distances = new double[n];
            double maxDist = 0;

            for (int j = 0; j < n; j++) {
                double minDistToCenter = Double.POSITIVE_INFINITY;
                for (int c = 0; c < i; c++) {
                    double dist = euclideanDistance(data[j], centers[c]);
                    minDistToCenter = Math.min(minDistToCenter, dist);
                }
                distances[j] = minDistToCenter;
                maxDist += distances[j] * distances[j];
            }

            // 按概率选择下一个中心
            double threshold = random.nextDouble() * maxDist;
            double cumSum = 0;

            for (int j = 0; j < n; j++) {
                cumSum += distances[j] * distances[j];
                if (cumSum >= threshold) {
                    centers[i] = data[j].clone();
                    break;
                }
            }
        }

        return centers;
    }

    /**
     * 随机初始化
     */
    private double[][] initRandom(double[][] data, int k) {
        int n = data.length;
        Random random = new Random(42);
        double[][] centers = new double[k][];

        Set<Integer> selected = new HashSet<>();
        for (int i = 0; i < k; i++) {
            int idx;
            do {
                idx = random.nextInt(n);
            } while (selected.contains(idx));

            selected.add(idx);
            centers[i] = data[idx].clone();
        }

        return centers;
    }

    /**
     * 获取聚类中心
     */
    public double[][] getCenters() {
        return centers;
    }

    /**
     * 获取惯性
     */
    public double getInertia() {
        return inertia[0];
    }

    @Override
    public int getNCluster() {
        return nClusters;
    }

    /**
     * 设置初始化方法
     */
    public KMeans setInitMethod(String method) {
        this.initMethod = method;
        return this;
    }

    /**
     * 设置初始化次数
     */
    public KMeans setNInit(int nInit) {
        this.nInit = nInit;
        return this;
    }
}
