package org.bytedeco.pytorch.serving.tritonserver.enums;

import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonInvalidArgumentException;

/** {@code TRITONSERVER_ModelControlMode} / Python {@code ModelControlMode}. */
public enum ModelControlMode {
    NONE(0),
    POLL(1),
    EXPLICIT(2);

    private final int code;

    ModelControlMode(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static ModelControlMode fromCode(int code) {
        for (ModelControlMode m : values()) {
            if (m.code == code) {
                return m;
            }
        }
        throw new TritonInvalidArgumentException("Unknown ModelControlMode code: " + code);
    }
}
