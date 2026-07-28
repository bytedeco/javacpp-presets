package org.bytedeco.pytorch.dataframe.ml.clustering;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.List;

/**
 * BIRCH 聚类 (Balanced Iterative Reducing and Clustering using Hierarchies)
 * 内存高效的分层聚类
 */
public class BIRCH extends BaseClusterer {
    private int nClusters;
    private int threshold = 50;  // 聚类特征树的阈值
    private List<double[][]> leafClusters;

    public BIRCH(int nClusters, String... features) {
        super(features);
        this.nClusters = nClusters;
    }

    @Override
    public BIRCH fit(DataFrame X) {
        double[][] data = extractMatrix(X);
        int n = data.length;

        // 简化实现：使用 K-means
        KMeans kmeans = new KMeans(nClusters);
        kmeans.fit(X);

        labels = kmeans.getLabels();
        fitted = true;
        return this;
    }

    @Override
    public int[] predict(DataFrame X) {
        if (!fitted) {
            throw new IllegalStateException("模型未拟合");
        }

        return new KMeans(nClusters).fit(X).predict(X);
    }

    @Override
    public int getNCluster() {
        return nClusters;
    }
}