package org.bytedeco.pytorch.dataframe.ml.clustering;

import org.bytedeco.pytorch.dataframe.DataFrame;
import java.util.*;

/**
 * DBSCAN 密度聚类
 * 基于密度的空间聚类算法
 * 可以发现任意形状的聚类，识别离群点
 */
public class DBSCAN extends BaseClusterer {
    private double eps;
    private int minSamples = 5;
    private int nClusters = 0;
    private Map<Integer, List<Integer>> neighbors;
    private double[][] trainData;  // 保存训练数据用于预测
    private int trainDataSize;

    public DBSCAN(double eps, String... features) {
        super(features);
        this.eps = eps;
    }

    public DBSCAN(double eps, int minSamples, String... features) {
        super(features);
        this.eps = eps;
        this.minSamples = minSamples;
    }

    @Override
    public DBSCAN fit(DataFrame X) {
        trainData = extractMatrix(X);
        trainDataSize = trainData.length;
        int n = trainDataSize;

        // 计算邻域
        neighbors = new HashMap<>();
        for (int i = 0; i < n; i++) {
            neighbors.put(i, new ArrayList<>());
            for (int j = 0; j < n; j++) {
                if (i != j && euclideanDistance(trainData[i], trainData[j]) <= eps) {
                    neighbors.get(i).add(j);
                }
            }
        }

        // DBSCAN 聚类
        labels = new int[n];
        Arrays.fill(labels, -1);  // -1 表示未访问或噪声

        int clusterId = 0;
        Set<Integer> visited = new HashSet<>();

        for (int i = 0; i < n; i++) {
            if (visited.contains(i)) {
                continue;
            }

            List<Integer> neighborList = neighbors.get(i);

            // 核心点：邻域内至少有 minSamples 个点
            if (neighborList.size() >= minSamples) {
                expandCluster(i, clusterId, visited);
                clusterId++;
            }
        }

        nClusters = clusterId;
        fitted = true;
        return this;
    }

    /**
     * 扩展聚类
     */
    private void expandCluster(int point, int clusterId, Set<Integer> visited) {
        Queue<Integer> queue = new LinkedList<>();
        queue.add(point);
        visited.add(point);
        labels[point] = clusterId;

        while (!queue.isEmpty()) {
            int current = queue.poll();
            List<Integer> neighborList = neighbors.get(current);

            // 只有核心点才能扩展聚类
            if (neighborList.size() >= minSamples) {
                for (int neighbor : neighborList) {
                    if (!visited.contains(neighbor)) {
                        visited.add(neighbor);
                        queue.add(neighbor);
                        labels[neighbor] = clusterId;
                    } else if (labels[neighbor] == -1) {
                        // 将边界点分配给聚类
                        labels[neighbor] = clusterId;
                    }
                }
            }
        }
    }

    @Override
    public int[] predict(DataFrame X) {
        if (!fitted) {
            throw new IllegalStateException("模型未拟合");
        }

        double[][] newData = extractMatrix(X);
        int n = newData.length;
        int[] predictions = new int[n];

        // 对新样本进行预测
        for (int i = 0; i < n; i++) {
            predictions[i] = predictSingleSample(newData[i]);
        }

        return predictions;
    }

    /**
     * 预测单个样本的聚类
     * 策略：找最近的训练样本，返回其聚类标签
     */
    private int predictSingleSample(double[] sample) {
        double minDist = Double.POSITIVE_INFINITY;
        int nearestTrainingSampleCluster = -1;

        // 找最近的训练样本
        for (int i = 0; i < trainDataSize; i++) {
            double dist = euclideanDistance(sample, trainData[i]);
            if (dist < minDist) {
                minDist = dist;
                nearestTrainingSampleCluster = labels[i];
            }
        }

        // 如果最近的训练样本是噪声点 (-1)，则新样本也被分类为噪声
        return nearestTrainingSampleCluster;
    }

    @Override
    public int getNCluster() {
        return nClusters;
    }

    /**
     * 设置邻域半径
     */
    public DBSCAN setEps(double eps) {
        this.eps = eps;
        return this;
    }

    /**
     * 设置最小样本数
     */
    public DBSCAN setMinSamples(int minSamples) {
        this.minSamples = minSamples;
        return this;
    }

    /**
     * 获取邻域信息
     */
    public Map<Integer, List<Integer>> getNeighbors() {
        return new HashMap<>(neighbors);
    }
}