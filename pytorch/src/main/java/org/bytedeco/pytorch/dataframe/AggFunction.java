package org.bytedeco.pytorch.dataframe;

/** Aggregation functions for pandas-style groupby. */
public enum AggFunction {
    SUM,
    MEAN,
    MEDIAN,
    MAX,
    MIN,
    COUNT,
    STD,       // sample std (ddof=1)
    VAR,       // sample variance (ddof=1)
    FIRST,
    LAST,
    NUNIQUE,   // count of unique values
    QUANTILE,  // requires extra param — handled specially
    MODE,
    SKEW,
    KURT       // excess kurtosis (Fisher's definition)
}
