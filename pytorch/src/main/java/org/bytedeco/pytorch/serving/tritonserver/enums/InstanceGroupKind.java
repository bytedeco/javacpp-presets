package org.bytedeco.pytorch.serving.tritonserver.enums;

import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonInvalidArgumentException;

/** {@code TRITONSERVER_InstanceGroupKind} / Python {@code InstanceGroupKind}. */
public enum InstanceGroupKind {
    AUTO(0),
    CPU(1),
    GPU(2),
    MODEL(3);

    private final int code;

    InstanceGroupKind(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static InstanceGroupKind fromCode(int code) {
        for (InstanceGroupKind k : values()) {
            if (k.code == code) {
                return k;
            }
        }
        throw new TritonInvalidArgumentException("Unknown InstanceGroupKind code: " + code);
    }
}
