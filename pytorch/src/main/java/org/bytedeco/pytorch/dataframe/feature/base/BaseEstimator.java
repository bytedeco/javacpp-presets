package org.bytedeco.pytorch.dataframe.feature.base;

import java.util.LinkedHashMap;
import java.util.Map;

/** sklearn-style estimator parameter API. */
public interface BaseEstimator {
    Map<String, Object> getParams();

    void setParams(Map<String, Object> params);

    default BaseEstimator cloneEstimator() {
        throw new UnsupportedOperationException(
            "cloneEstimator() not implemented for " + getClass().getSimpleName());
    }

    default Map<String, Object> emptyParams() {
        return new LinkedHashMap<>();
    }
}
