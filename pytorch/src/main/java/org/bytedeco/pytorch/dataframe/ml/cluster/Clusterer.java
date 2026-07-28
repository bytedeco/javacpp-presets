package org.bytedeco.pytorch.dataframe.ml.cluster;

import java.util.Map;

/**
 * Unified clustering interface mirroring sklearn's ClusterMixin + BaseEstimator.
 */
public interface Clusterer {
    /** Train on X (n_samples × n_features) */
    void fit(double[][] X);

    /** Predict cluster labels for X */
    int[] predict(double[][] X);

    /** Fit and return labels_ (convenience) */
    default int[] fitPredict(double[][] X) {
        fit(X);
        return getLabels();
    }

    Map<String, Object> getParams();
    void setParams(Map<String, Object> p);

    // Standard getters for sklearn-like attributes
    int[] getLabels();
    double[][] getClusterCenters();
    int getNClusters();
}

