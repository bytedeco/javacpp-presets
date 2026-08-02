package org.bytedeco.pytorch.serving.tritonserver.enums;

import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonInvalidArgumentException;

/** {@code TRITONSERVER_MetricKind} / Python {@code MetricKind}. */
public enum TritonMetricKind {
    COUNTER(0),
    GAUGE(1);

    private final int code;

    TritonMetricKind(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static TritonMetricKind fromCode(int code) {
        for (TritonMetricKind k : values()) {
            if (k.code == code) {
                return k;
            }
        }
        throw new TritonInvalidArgumentException("Unknown MetricKind code: " + code);
    }
}
