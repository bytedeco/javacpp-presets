package org.bytedeco.pytorch.dataframe.ml.clustering;

import org.bytedeco.pytorch.dataframe.DataFrame;
import java.util.*;

/**
 * OPTICS 聚类
 * 有序点对聚类分析
 * 克服 DBSCAN 对参数敏感的问题
 */
public class OPTICS extends BaseClusterer {
    private double maxEps;
    private int minSamples = 5;
    private double[] reachability;
    private int[] order;
    private int[] labels;
    private int nClusters = 0;

    public OPTICS(double maxEps, String... features) {
        super(features);
        this.maxEps = maxEps;
    }

    public OPTICS(double maxEps, int minSamples, String... features) {
        super(features);
        this.maxEps = maxEps;
        this.minSamples = minSamples;
    }

    @Override
    public OPTICS fit(DataFrame X) {
        double[][] data = extractMatrix(X);
        int n = data.length;

        // 初始化可达距离
        reachability = new double[n];
        Arrays.fill(reachability, Double.POSITIVE_INFINITY);

        // 计算核心距离
        double[] coreDistances = new double[n];
        for (int i = 0; i < n; i++) {
            List<Double> distances = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    distances.add(euclideanDistance(data[i], data[j]));
                }
            }
            Collections.sort(distances);
            if (distances.size() >= minSamples) {
                coreDistances[i] = distances.get(minSamples - 1);
            } else {
                coreDistances[i] = Double.POSITIVE_INFINITY;
            }
        }

        // OPTICS 排序
        order = new int[n];
        boolean[] processed = new boolean[n];
        PriorityQueue<Integer> seeds = new PriorityQueue<>(Comparator.comparingDouble(a -> reachability[a]));

        int idx = 0;
        for (int i = 0; i < n; i++) {
            if (!processed[i]) {
                if (coreDistances[i] != Double.POSITIVE_INFINITY) {
                    // 扩展聚类
                    seeds.clear();
                    seeds.add(i);
                    processed[i] = true;
                    order[idx++] = i;

                    while (!seeds.isEmpty()) {
                        int current = seeds.poll();

                        // 找邻域
                        List<Integer> neighbors = new ArrayList<>();
                        for (int j = 0; j < n; j++) {
                            if (!processed[j] && euclideanDistance(data[current], data[j]) <= maxEps) {
                                neighbors.add(j);
                            }
                        }

                        // 更新可达距离
                        for (int neighbor : neighbors) {
                            double reachDist = Math.max(
                                coreDistances[current],
                                euclideanDistance(data[current], data[neighbor])
                            );

                            if (reachDist < reachability[neighbor]) {
                                reachability[neighbor] = reachDist;
                                seeds.add(neighbor);
                            }

                            if (!processed[neighbor]) {
                                processed[neighbor] = true;
                                order[idx++] = neighbor;
                            }
                        }
                    }
                } else {
                    // 孤立点
                    processed[i] = true;
                    order[idx++] = i;
                }
            }
        }

        // 提取聚类
        labels = new int[n];
        Arrays.fill(labels, -1);

        nClusters = 0;
        double threshold = maxEps * 0.5;  // 简化阈值

        for (int i = 0; i < n; i++) {
            if (reachability[order[i]] <= threshold) {
                if (i == 0 || reachability[order[i - 1]] > threshold) {
                    nClusters++;
                }
                labels[order[i]] = nClusters - 1;
            }
        }

        fitted = true;
        return this;
    }

    @Override
    public int[] predict(DataFrame X) {
        throw new UnsupportedOperationException("OPTICS 不支持预测");
    }

    @Override
    public int getNCluster() {
        return nClusters;
    }

    /**
     * 获取可达距离
     */
    public double[] getReachability() {
        return reachability;
    }

    /**
     * 获取排序顺序
     */
    public int[] getOrder() {
        return order;
    }
}