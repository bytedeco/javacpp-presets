package org.bytedeco.pytorch.serving.tritonserver.enums;

import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonInvalidArgumentException;

/** {@code TRITONSERVER_MetricFormat} / Python {@code MetricFormat}. */
public enum TritonMetricFormat {
    PROMETHEUS(0);

    private final int code;

    TritonMetricFormat(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static TritonMetricFormat fromCode(int code) {
        for (TritonMetricFormat f : values()) {
            if (f.code == code) {
                return f;
            }
        }
        throw new TritonInvalidArgumentException("Unknown MetricFormat code: " + code);
    }
}
