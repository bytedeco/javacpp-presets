package org.bytedeco.pytorch.dataframe.ml.clustering;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * 聚类评估指标
 * 包含多种评估聚类质量的指标
 */
public class ClusteringMetrics {

    /**
     * 轮廓系数 (Silhouette Score)
     * 范围：[-1, 1]，越接近 1 越好
     * 衡量样本与自身聚类的相似度相对于其他聚类的相似度
     */
    public static double silhouetteScore(double[][] data, int[] labels) {
        int n = data.length;
        if (n < 2) return 0;

        int nClusters = Arrays.stream(labels).max().orElse(0) + 1;
        if (nClusters < 2) return 0;

        double totalScore = 0;

        for (int i = 0; i < n; i++) {
            int clusterI = labels[i];

            // 计算 a(i) - 同簇内其他点的平均距离
            double a_i = 0;
            int countA = 0;
            for (int j = 0; j < n; j++) {
                if (i != j && labels[j] == clusterI) {
                    a_i += euclideanDistance(data[i], data[j]);
                    countA++;
                }
            }
            if (countA > 0) {
                a_i /= countA;
            } else {
                a_i = 0;
            }

            // 计算 b(i) - 其他聚类内点的平均最小距离
            double b_i = Double.POSITIVE_INFINITY;
            for (int k = 0; k < nClusters; k++) {
                if (k != clusterI) {
                    double avgDist = 0;
                    int countB = 0;
                    for (int j = 0; j < n; j++) {
                        if (labels[j] == k) {
                            avgDist += euclideanDistance(data[i], data[j]);
                            countB++;
                        }
                    }
                    if (countB > 0) {
                        avgDist /= countB;
                        b_i = Math.min(b_i, avgDist);
                    }
                }
            }

            // 计算轮廓系数
            double s_i;
            if (Math.max(a_i, b_i) == 0) {
                s_i = 0;
            } else {
                s_i = (b_i - a_i) / Math.max(a_i, b_i);
            }

            totalScore += s_i;
        }

        return totalScore / n;
    }

    /**
     * Davies-Bouldin 指数
     * 范围：[0, ∞]，越小越好
     * 衡量聚类的分离度
     */
    public static double daviesBouldinIndex(double[][] data, int[] labels) {
        int nClusters = Arrays.stream(labels).max().orElse(0) + 1;
        if (nClusters < 2) return 0;

        double[][] centers = computeClusterCenters(data, labels, nClusters);
        double[] avgDistances = computeAverageDistances(data, labels, centers, nClusters);

        double totalScore = 0;
        for (int i = 0; i < nClusters; i++) {
            double maxRatio = 0;
            for (int j = 0; j < nClusters; j++) {
                if (i != j) {
                    double dist = euclideanDistance(centers[i], centers[j]);
                    double ratio = (avgDistances[i] + avgDistances[j]) / Math.max(dist, 1e-10);
                    maxRatio = Math.max(maxRatio, ratio);
                }
            }
            totalScore += maxRatio;
        }

        return totalScore / nClusters;
    }

    /**
     * Calinski-Harabasz 指数
     * 也称为方差比准则
     * 范围：[0, ∞]，越大越好
     */
    public static double calinskiHarabaszIndex(double[][] data, int[] labels) {
        int n = data.length;
        int nClusters = Arrays.stream(labels).max().orElse(0) + 1;
        if (nClusters < 2 || nClusters >= n) return 0;

        // 计算全局中心
        double[] globalCenter = computeGlobalCenter(data);

        // 类间方差 (Between-cluster variance)
        double[][] centers = computeClusterCenters(data, labels, nClusters);
        double betweenVar = 0;
        for (int k = 0; k < nClusters; k++) {
            int clusterSize = 0;
            for (int i = 0; i < n; i++) {
                if (labels[i] == k) clusterSize++;
            }
            double dist = euclideanDistance(centers[k], globalCenter);
            betweenVar += clusterSize * dist * dist;
        }
        betweenVar /= (nClusters - 1);

        // 类内方差 (Within-cluster variance)
        double withinVar = 0;
        for (int i = 0; i < n; i++) {
            double dist = euclideanDistance(data[i], centers[labels[i]]);
            withinVar += dist * dist;
        }
        withinVar /= (n - nClusters);

        return (betweenVar / Math.max(withinVar, 1e-10)) * ((n - nClusters) / (nClusters - 1));
    }

    /**
     * 同质性 (Homogeneity)
     * 每个聚类只包含单一类别的样本
     */
    public static double homogeneity(int[] labels, int[] trueLabels) {
        int n = labels.length;
        if (n < 2) return 0;

        // 计算条件熵
        int nClusters = Arrays.stream(labels).max().orElse(0) + 1;
        int nClasses = Arrays.stream(trueLabels).max().orElse(0) + 1;

        // H(C|K) - 给定聚类条件下的真实标签熵
        double conditionalEntropy = 0;
        for (int k = 0; k < nClusters; k++) {
            int clusterSize = 0;
            Map<Integer, Integer> classCounts = new HashMap<>();

            for (int i = 0; i < n; i++) {
                if (labels[i] == k) {
                    clusterSize++;
                    classCounts.put(trueLabels[i], classCounts.getOrDefault(trueLabels[i], 0) + 1);
                }
            }

            if (clusterSize > 0) {
                double clusterEntropy = 0;
                for (int count : classCounts.values()) {
                    double p = (double) count / clusterSize;
                    if (p > 0) {
                        clusterEntropy -= p * Math.log(p);
                    }
                }
                conditionalEntropy += (clusterSize / (double) n) * clusterEntropy;
            }
        }

        // H(C) - 真实标签的边界熵
        double classEntropy = 0;
        int[] classCounts = new int[nClasses];
        for (int label : trueLabels) {
            classCounts[label]++;
        }
        for (int count : classCounts) {
            if (count > 0) {
                double p = (double) count / n;
                classEntropy -= p * Math.log(p);
            }
        }

        if (classEntropy < 1e-10) return 0;

        return 1.0 - (conditionalEntropy / classEntropy);
    }

    /**
     * 完整性 (Completeness)
     * 真实类别的所有样本都在同一聚类中
     */
    public static double completeness(int[] labels, int[] trueLabels) {
        int n = labels.length;
        if (n < 2) return 0;

        int nClusters = Arrays.stream(labels).max().orElse(0) + 1;
        int nClasses = Arrays.stream(trueLabels).max().orElse(0) + 1;

        // H(K|C) - 给定真实标签条件下的聚类熵
        double conditionalEntropy = 0;
        for (int c = 0; c < nClasses; c++) {
            int classSize = 0;
            Map<Integer, Integer> clusterCounts = new HashMap<>();

            for (int i = 0; i < n; i++) {
                if (trueLabels[i] == c) {
                    classSize++;
                    clusterCounts.put(labels[i], clusterCounts.getOrDefault(labels[i], 0) + 1);
                }
            }

            if (classSize > 0) {
                double classEntropy = 0;
                for (int count : clusterCounts.values()) {
                    double p = (double) count / classSize;
                    if (p > 0) {
                        classEntropy -= p * Math.log(p);
                    }
                }
                conditionalEntropy += (classSize / (double) n) * classEntropy;
            }
        }

        // H(K) - 聚类的边界熵
        double clusterEntropy = 0;
        int[] clusterCounts = new int[nClusters];
        for (int label : labels) {
            clusterCounts[label]++;
        }
        for (int count : clusterCounts) {
            if (count > 0) {
                double p = (double) count / n;
                clusterEntropy -= p * Math.log(p);
            }
        }

        if (clusterEntropy < 1e-10) return 0;

        return 1.0 - (conditionalEntropy / clusterEntropy);
    }

    /**
     * V 测度 (V-measure)
     * 同质性和完整性的调和平均
     */
    public static double vMeasure(int[] labels, int[] trueLabels) {
        double h = homogeneity(labels, trueLabels);
        double c = completeness(labels, trueLabels);

        if (h + c == 0) return 0;

        return 2.0 * (h * c) / (h + c);
    }

    /**
     * 互信息 (Mutual Information)
     */
    public static double mutualInfo(int[] labels, int[] trueLabels) {
        int n = labels.length;
        int nClusters = Arrays.stream(labels).max().orElse(0) + 1;
        int nClasses = Arrays.stream(trueLabels).max().orElse(0) + 1;

        double mi = 0;
        for (int k = 0; k < nClusters; k++) {
            for (int c = 0; c < nClasses; c++) {
                int count = 0;
                for (int i = 0; i < n; i++) {
                    if (labels[i] == k && trueLabels[i] == c) {
                        count++;
                    }
                }

                if (count > 0) {
                    double p_kc = (double) count / n;

                    int clusterSize = 0, classSize = 0;
                    for (int i = 0; i < n; i++) {
                        if (labels[i] == k) clusterSize++;
                        if (trueLabels[i] == c) classSize++;
                    }

                    double p_k = (double) clusterSize / n;
                    double p_c = (double) classSize / n;

                    mi += p_kc * Math.log(p_kc / (p_k * p_c));
                }
            }
        }

        return mi;
    }

    /**
     * 调整兰德指数 (Adjusted Rand Index)
     * 范围：[-1, 1]，越接近 1 越好
     */
    public static double adjustedRandIndex(int[] labels, int[] trueLabels) {
        int n = labels.length;

        // 计算聚类对的一致性
        long sameClusterSameClass = 0, sameClusterDiffClass = 0;
        long diffClusterSameClass = 0, diffClusterDiffClass = 0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                boolean sameCluster = labels[i] == labels[j];
                boolean sameClass = trueLabels[i] == trueLabels[j];

                if (sameCluster && sameClass) {
                    sameClusterSameClass++;
                } else if (sameCluster && !sameClass) {
                    sameClusterDiffClass++;
                } else if (!sameCluster && sameClass) {
                    diffClusterSameClass++;
                } else {
                    diffClusterDiffClass++;
                }
            }
        }

        long TP = sameClusterSameClass;
        long FP = sameClusterDiffClass;
        long FN = diffClusterSameClass;
        long TN = diffClusterDiffClass;

        long pairs = TP + FP + FN + TN;
        if (pairs == 0) return 0;

        double ri = (double) (TP + TN) / pairs;

        double e = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (double) (pairs * pairs);

        if (Math.abs(1 - e) < 1e-10) return 0;

        return (ri - e) / (1 - e);
    }

    /**
     * 纯度 (Purity)
     * 范围：[0, 1]，越接近 1 越好
     */
    public static double purity(int[] labels, int[] trueLabels) {
        int n = labels.length;
        if (n == 0) return 0;

        int nClusters = Arrays.stream(labels).max().orElse(0) + 1;

        int correctCount = 0;
        for (int k = 0; k < nClusters; k++) {
            Map<Integer, Integer> classCounts = new HashMap<>();
            int clusterSize = 0;

            for (int i = 0; i < n; i++) {
                if (labels[i] == k) {
                    clusterSize++;
                    classCounts.put(trueLabels[i], classCounts.getOrDefault(trueLabels[i], 0) + 1);
                }
            }

            if (clusterSize > 0) {
                int maxCount = classCounts.values().stream().max(Integer::compare).orElse(0);
                correctCount += maxCount;
            }
        }

        return (double) correctCount / n;
    }

    // ============ 辅助方法 ============

    private static double euclideanDistance(double[] p1, double[] p2) {
        double sum = 0;
        for (int i = 0; i < p1.length; i++) {
            double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    private static double[] computeGlobalCenter(double[][] data) {
        int n = data.length;
        int d = data[0].length;
        double[] center = new double[d];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                center[j] += data[i][j];
            }
        }

        for (int j = 0; j < d; j++) {
            center[j] /= n;
        }

        return center;
    }

    private static double[][] computeClusterCenters(double[][] data, int[] labels, int nClusters) {
        int d = data[0].length;
        double[][] centers = new double[nClusters][d];
        int[] clusterSizes = new int[nClusters];

        for (int i = 0; i < data.length; i++) {
            int cluster = labels[i];
            clusterSizes[cluster]++;
            for (int j = 0; j < d; j++) {
                centers[cluster][j] += data[i][j];
            }
        }

        for (int k = 0; k < nClusters; k++) {
            if (clusterSizes[k] > 0) {
                for (int j = 0; j < d; j++) {
                    centers[k][j] /= clusterSizes[k];
                }
            }
        }

        return centers;
    }

    private static double[] computeAverageDistances(double[][] data, int[] labels,
                                                    double[][] centers, int nClusters) {
        double[] avgDistances = new double[nClusters];
        int[] clusterSizes = new int[nClusters];

        for (int i = 0; i < data.length; i++) {
            int cluster = labels[i];
            clusterSizes[cluster]++;
            avgDistances[cluster] += euclideanDistance(data[i], centers[cluster]);
        }

        for (int k = 0; k < nClusters; k++) {
            if (clusterSizes[k] > 0) {
                avgDistances[k] /= clusterSizes[k];
            }
        }

        return avgDistances;
    }
}