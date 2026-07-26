package org.bytedeco.pytorch.data.dataframe.enums;

/** Power / quantile transform methods. */
public enum TransformMethod {
    YEO_JOHNSON, BOX_COX
}

/** Quantile transform output distribution. */
enum QuantileOutput {
    UNIFORM, NORMAL
}
