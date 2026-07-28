package org.bytedeco.pytorch.dataframe.ml.clustering;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.HashSet;
import java.util.Set;

/**
 * 亲和传播聚类 (Affinity Propagation)
 * 基于消息传递的聚类算法
 */
public class AffinityPropagation extends BaseClusterer {
    private int maxIter = 200;
    private double damping = 0.9;
    private double[][] similarities;
    private int nClusters = 0;

    public AffinityPropagation(String... features) {
        super(features);
    }

    @Override
    public AffinityPropagation fit(DataFrame X) {
        double[][] data = extractMatrix(X);
        int n = data.length;

        // 计算相似度矩阵
        similarities = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    // 自相似度（偏好）
                    similarities[i][j] = -euclideanDistance(data[i], data[j]);
                } else {
                    similarities[i][j] = -euclideanDistance(data[i], data[j]);
                }
            }
        }

        // 初始化信息矩阵
        double[][] responsibility = new double[n][n];
        double[][] availability = new double[n][n];

        // AP 迭代
        for (int iter = 0; iter < maxIter; iter++) {
            // 更新 responsibility
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    double max = -Double.MAX_VALUE;
                    for (int j = 0; j < n; j++) {
                        if (j != k) {
                            max = Math.max(max, availability[i][j] + similarities[i][j]);
                        }
                    }
                    responsibility[i][k] = similarities[i][k] - max;
                }
            }

            // 更新 availability
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    if (i == k) {
                        double sum = 0;
                        for (int j = 0; j < n; j++) {
                            if (j != k) {
                                sum += Math.max(0, responsibility[j][k]);
                            }
                        }
                        availability[k][k] = sum;
                    } else {
                        double sum = Math.max(0, responsibility[k][k]);
                        for (int j = 0; j < n; j++) {
                            if (j != i && j != k) {
                                sum += Math.max(0, responsibility[j][k]);
                            }
                        }
                        availability[i][k] = Math.min(0, sum);
                    }
                }
            }

            // 阻尼处理
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    responsibility[i][k] = damping * responsibility[i][k] + 
                                         (1 - damping) * (similarities[i][k] - 
                                         Math.max(0, availability[i][k]));
                }
            }
        }

        // 提取聚类
        labels = new int[n];
        Set<Integer> exemplars = new HashSet<>();

        for (int i = 0; i < n; i++) {
            int maxK = 0;
            double maxScore = responsibility[i][0] + availability[i][0];

            for (int k = 1; k < n; k++) {
                double score = responsibility[i][k] + availability[i][k];
                if (score > maxScore) {
                    maxScore = score;
                    maxK = k;
                }
            }

            labels[i] = maxK;
            exemplars.add(maxK);
        }

        nClusters = exemplars.size();
        fitted = true;
        return this;
    }

    @Override
    public int[] predict(DataFrame X) {
        throw new UnsupportedOperationException("亲和传播不支持预测");
    }

    @Override
    public int getNCluster() {
        return nClusters;
    }

    /**
     * 设置迭代次数
     */
    public AffinityPropagation setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    /**
     * 设置阻尼系数
     */
    public AffinityPropagation setDamping(double damping) {
        this.damping = damping;
        return this;
    }
}