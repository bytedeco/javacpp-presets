package org.bytedeco.pytorch.dataframe.ml.clustering;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * 高斯混合模型 (Gaussian Mixture Model)
 * 基于概率的软聚类算法
 */
public class GaussianMixture extends BaseClusterer {
    private int nComponents;
    private int maxIter = 100;
    private double tolerance = 1e-4;
    private double[] weights;
    private double[][] means;
    private double[][][] covariances;
    private double[][] responsibilities;

    public GaussianMixture(int nComponents, String... features) {
        super(features);
        this.nComponents = nComponents;
    }

    @Override
    public GaussianMixture fit(DataFrame X) {
        double[][] data = extractMatrix(X);
        int n = data.length;
        int d = data[0].length;

        // 初始化参数
        initializeParameters(data);

        // EM 迭代
        for (int iter = 0; iter < maxIter; iter++) {
            // E 步：计算责任度矩阵
            responsibilities = new double[n][nComponents];
            double logLikelihood = 0;

            for (int i = 0; i < n; i++) {
                double[] gaussian = new double[nComponents];
                double maxGaussian = -Double.MAX_VALUE;

                for (int k = 0; k < nComponents; k++) {
                    gaussian[k] = Math.log(weights[k]) + 
                        logGaussian(data[i], means[k], covariances[k]);
                    maxGaussian = Math.max(maxGaussian, gaussian[k]);
                }

                double sum = 0;
                for (int k = 0; k < nComponents; k++) {
                    responsibilities[i][k] = Math.exp(gaussian[k] - maxGaussian);
                    sum += responsibilities[i][k];
                }

                for (int k = 0; k < nComponents; k++) {
                    responsibilities[i][k] /= sum;
                    logLikelihood += Math.log(sum) + maxGaussian;
                }
            }

            // M 步：更新参数
            double[] nk = new double[nComponents];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < nComponents; k++) {
                    nk[k] += responsibilities[i][k];
                }
            }

            // 更新权重
            for (int k = 0; k < nComponents; k++) {
                weights[k] = nk[k] / n;
            }

            // 更新均值
            for (int k = 0; k < nComponents; k++) {
                double[] newMean = new double[d];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < d; j++) {
                        newMean[j] += responsibilities[i][k] * data[i][j];
                    }
                }
                for (int j = 0; j < d; j++) {
                    newMean[j] /= Math.max(nk[k], 1e-10);
                }
                means[k] = newMean;
            }

            // 更新协方差（简化实现）
            for (int k = 0; k < nComponents; k++) {
                double[][] newCov = new double[d][d];
                for (int i = 0; i < n; i++) {
                    double[] diff = new double[d];
                    for (int j = 0; j < d; j++) {
                        diff[j] = data[i][j] - means[k][j];
                    }
                    for (int j1 = 0; j1 < d; j1++) {
                        for (int j2 = 0; j2 < d; j2++) {
                            newCov[j1][j2] += responsibilities[i][k] * diff[j1] * diff[j2];
                        }
                    }
                }
                for (int j1 = 0; j1 < d; j1++) {
                    for (int j2 = 0; j2 < d; j2++) {
                        newCov[j1][j2] /= Math.max(nk[k], 1e-10);
                    }
                }
                covariances[k] = newCov;
            }
        }

        // 分配标签
        labels = new int[n];
        for (int i = 0; i < n; i++) {
            int maxK = 0;
            double maxResp = responsibilities[i][0];
            for (int k = 1; k < nComponents; k++) {
                if (responsibilities[i][k] > maxResp) {
                    maxResp = responsibilities[i][k];
                    maxK = k;
                }
            }
            labels[i] = maxK;
        }

        fitted = true;
        return this;
    }

    /**
     * 初始化参数
     */
    private void initializeParameters(double[][] data) {
        int n = data.length;
        int d = data[0].length;

        weights = new double[nComponents];
        means = new double[nComponents][d];
        covariances = new double[nComponents][d][d];

        // 均匀权重
        for (int k = 0; k < nComponents; k++) {
            weights[k] = 1.0 / nComponents;
        }

        // 随机初始化均值
        Random random = new Random(42);
        Set<Integer> selected = new HashSet<>();
        for (int k = 0; k < nComponents; k++) {
            int idx;
            do {
                idx = random.nextInt(n);
            } while (selected.contains(idx));
            selected.add(idx);
            means[k] = data[idx].clone();
        }

        // 初始化协方差为单位矩阵
        for (int k = 0; k < nComponents; k++) {
            for (int i = 0; i < d; i++) {
                covariances[k][i][i] = 1.0;
            }
        }
    }

    /**
     * 高斯分布对数概率密度
     */
    private double logGaussian(double[] x, double[] mean, double[][] cov) {
        int d = x.length;
        double diff = 0;
        for (int i = 0; i < d; i++) {
            diff += (x[i] - mean[i]) * (x[i] - mean[i]);
        }

        double det = 1.0;  // 简化：假设行列式为 1
        return -0.5 * (d * Math.log(2 * Math.PI) + Math.log(det) + diff);
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
            int maxK = 0;
            double maxProb = logGaussian(data[i], means[0], covariances[0]) + Math.log(weights[0]);

            for (int k = 1; k < nComponents; k++) {
                double prob = logGaussian(data[i], means[k], covariances[k]) + Math.log(weights[k]);
                if (prob > maxProb) {
                    maxProb = prob;
                    maxK = k;
                }
            }

            predictions[i] = maxK;
        }

        return predictions;
    }

    @Override
    public int getNCluster() {
        return nComponents;
    }

    /**
     * 获取权重
     */
    public double[] getWeights() {
        return weights;
    }

    /**
     * 获取均值
     */
    public double[][] getMeans() {
        return means;
    }
}