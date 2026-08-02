package org.bytedeco.pytorch.serving.tritonserver.enums;

import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonInvalidArgumentException;

/** {@code TRITONSERVER_LogFormat} / Python {@code LogFormat}. */
public enum TritonLogFormat {
    DEFAULT(0),
    ISO8601(1);

    private final int code;

    TritonLogFormat(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static TritonLogFormat fromCode(int code) {
        for (TritonLogFormat f : values()) {
            if (f.code == code) {
                return f;
            }
        }
        throw new TritonInvalidArgumentException("Unknown LogFormat code: " + code);
    }
}
