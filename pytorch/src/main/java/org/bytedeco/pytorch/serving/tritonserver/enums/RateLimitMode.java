package org.bytedeco.pytorch.serving.tritonserver.enums;

import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonInvalidArgumentException;

/** {@code TRITONSERVER_RateLimitMode} / Python {@code RateLimitMode}. */
public enum RateLimitMode {
    OFF(0),
    EXEC_COUNT(1);

    private final int code;

    RateLimitMode(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static RateLimitMode fromCode(int code) {
        for (RateLimitMode m : values()) {
            if (m.code == code) {
                return m;
            }
        }
        throw new TritonInvalidArgumentException("Unknown RateLimitMode code: " + code);
    }
}
