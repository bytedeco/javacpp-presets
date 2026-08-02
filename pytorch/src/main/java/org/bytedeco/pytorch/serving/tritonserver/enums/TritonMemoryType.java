package org.bytedeco.pytorch.serving.tritonserver.enums;

import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonInvalidArgumentException;

/**
 * Memory type for tensor buffers.
 *
 * <p>Values match {@code TRITONSERVER_MemoryType}.
 */
public enum TritonMemoryType {
    CPU(0),
    CPU_PINNED(1),
    GPU(2);

    private final int code;

    TritonMemoryType(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static TritonMemoryType fromCode(int code) {
        for (TritonMemoryType m : values()) {
            if (m.code == code) {
                return m;
            }
        }
        throw new TritonInvalidArgumentException("Unknown MemoryType code: " + code);
    }

    public String typeString() {
        return switch (this) {
            case CPU -> "CPU";
            case CPU_PINNED -> "CPU_PINNED";
            case GPU -> "GPU";
        };
    }
}
