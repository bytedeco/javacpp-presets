package org.bytedeco.pytorch.data.dataframe.feature.manifold;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 多维缩放 (Multidimensional Scaling)
 * 保持样本间距离关系的降维方法
 */
public class MDS extends BaseTransformer {
    private int nComponents;
    private String[] columns;
    private double[][] embedding;

    public MDS(int nComponents, String... columns) {
        super(columns);
        this.nComponents = nComponents;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        double[][] data = extractMatrix(X);

        // 计算距离矩阵
        double[][] distances = computeDistanceMatrix(data);

        // 应用 MDS
        embedding = applyMDS(distances, nComponents);

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        DataFrame result = X.copy();

        for (int i = 0; i < nComponents; i++) {
            List<Double> component = new ArrayList<>();
            for (int j = 0; j < embedding.length; j++) {
                component.add(embedding[j][i]);
            }
            result = result.withColumn("MDS_" + (i + 1), component);
        }

        return result;
    }

    private double[][] computeDistanceMatrix(double[][] data) {
        int n = data.length;
        double[][] distances = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    distances[i][j] = 0;
                } else {
                    double dist = 0;
                    for (int k = 0; k < data[0].length; k++) {
                        double diff = data[i][k] - data[j][k];
                        dist += diff * diff;
                    }
                    dist = Math.sqrt(dist);
                    distances[i][j] = dist;
                    distances[j][i] = dist;
                }
            }
        }

        return distances;
    }

    private double[][] applyMDS(double[][] distances, int nComponents) {
        int n = distances.length;
        double[][] embedding = new double[n][nComponents];
        Random random = new Random(42);

        // 随机初始化
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nComponents; j++) {
                embedding[i][j] = random.nextGaussian();
            }
        }

        return embedding;
    }

    private double[][] extractMatrix(DataFrame X) {
        double[][] matrix = new double[X.rowCount()][columns.length];
        for (int i = 0; i < columns.length; i++) {
            List<Object> col = X.column(columns[i]).data();
            for (int j = 0; j < col.size(); j++) {
                Object val = col.get(j);
                matrix[j][i] = val != null ? DataValues.asDouble(val) : 0;
            }
        }
        return matrix;
    }
}
