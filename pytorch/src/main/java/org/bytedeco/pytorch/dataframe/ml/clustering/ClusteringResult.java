package org.bytedeco.pytorch.dataframe.ml.clustering;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 聚类结果封装
 */
public class ClusteringResult {
    private int[] labels;
    private int nClusters;
    private double silhouetteScore;
    private Map<Integer, List<Integer>> clusters;
    private double[] clusterSizes;
    private double[][] clusterCenters;

    public ClusteringResult(int[] labels, int nClusters) {
        this.labels = labels;
        this.nClusters = nClusters;
        this.clusters = new HashMap<>();

        // 初始化聚类
        for (int i = 0; i < nClusters; i++) {
            clusters.put(i, new ArrayList<>());
        }

        // 分配样本到聚类
        for (int i = 0; i < labels.length; i++) {
            int label = labels[i];
            if (label >= 0) {  // -1 表示噪声点
                clusters.get(label).add(i);
            }
        }

        // 计算聚类大小
        clusterSizes = new double[nClusters];
        for (int i = 0; i < nClusters; i++) {
            clusterSizes[i] = clusters.get(i).size();
        }
    }

    /**
     * 添加到 DataFrame
     */
    public DataFrame addToDataFrame(DataFrame df) throws Exception {
        List<Integer> labelList = new ArrayList<>();
        for (int label : labels) {
            labelList.add(label);
        }
        return df.withColumn("cluster", labelList);
    }

    /**
     * 获取聚类成员
     */
    public List<Integer> getClusterMembers(int clusterId) {
        return clusters.getOrDefault(clusterId, new ArrayList<>());
    }

    /**
     * 获取聚类大小
     */
    public double[] getClusterSizes() {
        return clusterSizes;
    }

    /**
     * 获取标签
     */
    public int[] getLabels() {
        return labels;
    }

    /**
     * 获取聚类数
     */
    public int getNCluster() {
        return nClusters;
    }

    /**
     * 设置轮廓系数
     */
    public void setSilhouetteScore(double score) {
        this.silhouetteScore = score;
    }

    /**
     * 获取轮廓系数
     */
    public double getSilhouetteScore() {
        return silhouetteScore;
    }

    /**
     * 打印聚类统计
     */
    public void printStatistics() {
        System.out.println("聚类统计信息:");
        System.out.printf("  聚类数: %d%n", nClusters);
        System.out.printf("  样本总数: %d%n", labels.length);
        for (int i = 0; i < nClusters; i++) {
            System.out.printf("  聚类 %d: %d 个样本 (%.1f%%)%n",
                    i, (int) clusterSizes[i],
                    clusterSizes[i] / labels.length * 100);
        }
        if (silhouetteScore >= 0) {
            System.out.printf("  轮廓系数: %.4f%n", silhouetteScore);
        }
    }
}