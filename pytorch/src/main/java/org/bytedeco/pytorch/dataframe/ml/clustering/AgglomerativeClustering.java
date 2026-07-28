package org.bytedeco.pytorch.dataframe.ml.clustering;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 凝聚层次聚类 (Agglomerative Clustering)
 * 自下而上的分层聚类算法
 */
public class AgglomerativeClustering extends BaseClusterer {
    private int nClusters;
    private String linkage = "ward";  // ward, complete, average, single
    private double[][] distanceMatrix;
    private double[][] trainData;
    private int trainDataSize;

//    public AgglomerativeClustering(int nClusters, String... features) {
//        super(features);
//        this.nClusters = nClusters;
//    }

    public AgglomerativeClustering(int nClusters, String linkage, String... features) {
        super(features);
        this.nClusters = nClusters;
        this.linkage = linkage;
    }

    @Override
    public AgglomerativeClustering fit(DataFrame X) {
        trainData = extractMatrix(X);
        trainDataSize = trainData.length;
        int n = trainDataSize;

        // 计算距离矩阵
        distanceMatrix = computeDistanceMatrix(trainData);

        // 初始化聚类 - 每个样本自成一个聚类
        List<Set<Integer>> clusters = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Set<Integer> cluster = new HashSet<>();
            cluster.add(i);
            clusters.add(cluster);
        }

        // 合并聚类直到达到目标数量
        while (clusters.size() > nClusters) {
            // 找到最近的两个聚类
            double minDist = Double.POSITIVE_INFINITY;
            int cluster1 = 0, cluster2 = 1;

            for (int i = 0; i < clusters.size(); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    double dist = computeClusterDistance(
                            clusters.get(i), clusters.get(j), trainData
                    );
                    if (dist < minDist) {
                        minDist = dist;
                        cluster1 = i;
                        cluster2 = j;
                    }
                }
            }

            // 合并两个聚类
            Set<Integer> merged = new HashSet<>(clusters.get(cluster1));
            merged.addAll(clusters.get(cluster2));

            // 移除两个旧聚类（注意顺序，先删大索引）
            clusters.remove(Math.max(cluster1, cluster2));
            clusters.remove(Math.min(cluster1, cluster2));
            clusters.add(merged);
        }

        // 分配标签
        labels = new int[n];
        for (int clusterId = 0; clusterId < clusters.size(); clusterId++) {
            for (int sampleId : clusters.get(clusterId)) {
                labels[sampleId] = clusterId;
            }
        }

        fitted = true;
        return this;
    }

    /**
     * 计算两个聚类之间的距离
     */
    private double computeClusterDistance(Set<Integer> c1, Set<Integer> c2, double[][] data) {
        switch (linkage) {
            case "single":
                return singleLinkage(c1, c2);
            case "complete":
                return completeLinkage(c1, c2);
            case "average":
                return averageLinkage(c1, c2);
            case "ward":
                return wardLinkage(c1, c2, data);
            default:
                return completeLinkage(c1, c2);
        }
    }

    /**
     * 单链接 - 两个聚类间的最小距离
     */
    private double singleLinkage(Set<Integer> c1, Set<Integer> c2) {
        double minDist = Double.POSITIVE_INFINITY;
        for (int i : c1) {
            for (int j : c2) {
                minDist = Math.min(minDist, distanceMatrix[i][j]);
            }
        }
        return minDist;
    }

    /**
     * 完全链接 - 两个聚类间的最大距离
     */
    private double completeLinkage(Set<Integer> c1, Set<Integer> c2) {
        double maxDist = 0;
        for (int i : c1) {
            for (int j : c2) {
                maxDist = Math.max(maxDist, distanceMatrix[i][j]);
            }
        }
        return maxDist;
    }

    /**
     * 平均链接 - 两个聚类间的平均距离
     */
    private double averageLinkage(Set<Integer> c1, Set<Integer> c2) {
        double sumDist = 0;
        int count = 0;
        for (int i : c1) {
            for (int j : c2) {
                sumDist += distanceMatrix[i][j];
                count++;
            }
        }
        return sumDist / Math.max(count, 1);
    }

    /**
     * Ward 链接 - 基于方差最小化
     */
    private double wardLinkage(Set<Integer> c1, Set<Integer> c2, double[][] data) {
        double[] center1 = computeCenter(c1, data);
        double[] center2 = computeCenter(c2, data);
        double dist = euclideanDistance(center1, center2);
        return dist * dist;  // Ward 链接使用距离的平方
    }

    /**
     * 计算聚类中心
     */
    private double[] computeCenter(Set<Integer> cluster, double[][] data) {
        double[] center = new double[data[0].length];
        for (int idx : cluster) {
            for (int j = 0; j < data[0].length; j++) {
                center[j] += data[idx][j];
            }
        }
        for (int j = 0; j < center.length; j++) {
            center[j] /= cluster.size();
        }
        return center;
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
     * 设置连接方法
     */
    public AgglomerativeClustering setLinkage(String linkage) {
        this.linkage = linkage;
        return this;
    }
}
//package lance.clustering;
//
// import org.bytedeco.pytorch.dataframe.DataFrame;
//import java.util.*;
//
///**
// * 凝聚层次聚类 (Agglomerative Clustering)
// * 自下而上的分层聚类算法
// */


//public class AgglomerativeClustering extends BaseClusterer {
//    private int nClusters;
//    private String linkage = "ward";  // ward, complete, average, single
//    private double[][] distanceMatrix;
//
////    public AgglomerativeClustering(int nClusters, String... features) {
////        super(features);
////        this.nClusters = nClusters;
////    }
//
//    public AgglomerativeClustering(int nClusters, String linkage, String... features) {
//        super(features);
//        this.nClusters = nClusters;
//        this.linkage = linkage;
//    }
//
//    @Override
//    public AgglomerativeClustering fit(DataFrame X) {
//        double[][] data = extractMatrix(X);
//        int n = data.length;
//
//        // 计算距离矩阵
//        distanceMatrix = computeDistanceMatrix(data);
//
//        // 初始化聚类
//        List<Set<Integer>> clusters = new ArrayList<>();
//        for (int i = 0; i < n; i++) {
//            Set<Integer> cluster = new HashSet<>();
//            cluster.add(i);
//            clusters.add(cluster);
//        }
//
//        // 合并聚类直到达到目标数量
//        while (clusters.size() > nClusters) {
//            // 找到最近的两个聚类
//            double minDist = Double.POSITIVE_INFINITY;
//            int cluster1 = 0, cluster2 = 1;
//
//            for (int i = 0; i < clusters.size(); i++) {
//                for (int j = i + 1; j < clusters.size(); j++) {
//                    double dist = computeClusterDistance(
//                        clusters.get(i), clusters.get(j), data
//                    );
//                    if (dist < minDist) {
//                        minDist = dist;
//                        cluster1 = i;
//                        cluster2 = j;
//                    }
//                }
//            }
//
//            // 合并两个聚类
//            Set<Integer> merged = new HashSet<>(clusters.get(cluster1));
//            merged.addAll(clusters.get(cluster2));
//
//            clusters.remove(Math.max(cluster1, cluster2));
//            clusters.remove(Math.min(cluster1, cluster2));
//            clusters.add(merged);
//        }
//
//        // 分配标签
//        labels = new int[n];
//        for (int clusterId = 0; clusterId < clusters.size(); clusterId++) {
//            for (int sampleId : clusters.get(clusterId)) {
//                labels[sampleId] = clusterId;
//            }
//        }
//
//        fitted = true;
//        return this;
//    }
//
//    /**
//     * 计算两个聚类之间的距离
//     */
//    private double computeClusterDistance(Set<Integer> c1, Set<Integer> c2, double[][] data) {
//        switch (linkage) {
//            case "single":
//                return singleLinkage(c1, c2);
//            case "complete":
//                return completeLinkage(c1, c2);
//            case "average":
//                return averageLinkage(c1, c2);
//            case "ward":
//                return wardLinkage(c1, c2, data);
//            default:
//                return completeLinkage(c1, c2);
//        }
//    }
//
//    private double singleLinkage(Set<Integer> c1, Set<Integer> c2) {
//        double minDist = Double.POSITIVE_INFINITY;
//        for (int i : c1) {
//            for (int j : c2) {
//                minDist = Math.min(minDist, distanceMatrix[i][j]);
//            }
//        }
//        return minDist;
//    }
//
//    private double completeLinkage(Set<Integer> c1, Set<Integer> c2) {
//        double maxDist = 0;
//        for (int i : c1) {
//            for (int j : c2) {
//                maxDist = Math.max(maxDist, distanceMatrix[i][j]);
//            }
//        }
//        return maxDist;
//    }
//
//    private double averageLinkage(Set<Integer> c1, Set<Integer> c2) {
//        double sumDist = 0;
//        int count = 0;
//        for (int i : c1) {
//            for (int j : c2) {
//                sumDist += distanceMatrix[i][j];
//                count++;
//            }
//        }
//        return sumDist / count;
//    }
//
//    private double wardLinkage(Set<Integer> c1, Set<Integer> c2, double[][] data) {
//        // 简化实现：使用质心连接
//        double[] center1 = computeCenter(c1, data);
//        double[] center2 = computeCenter(c2, data);
//        return euclideanDistance(center1, center2);
//    }
//
//    private double[] computeCenter(Set<Integer> cluster, double[][] data) {
//        double[] center = new double[data[0].length];
//        for (int idx : cluster) {
//            for (int j = 0; j < data[0].length; j++) {
//                center[j] += data[idx][j];
//            }
//        }
//        for (int j = 0; j < center.length; j++) {
//            center[j] /= cluster.size();
//        }
//        return center;
//    }
//
//    @Override
//    public int[] predict(DataFrame X) {
//        throw new UnsupportedOperationException("分层聚类不支持预测");
//    }
//
//    @Override
//    public int getNCluster() {
//        return nClusters;
//    }
//
//    /**
//     * 设置连接方法
//     */
//    public AgglomerativeClustering setLinkage(String linkage) {
//        this.linkage = linkage;
//        return this;
//    }
//}