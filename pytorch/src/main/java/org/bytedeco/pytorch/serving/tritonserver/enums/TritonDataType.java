package org.bytedeco.pytorch.serving.tritonserver.enums;

import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonInvalidArgumentException;

/**
 * Triton tensor data type.
 *
 * <p>Codes match {@code TRITONSERVER_DataType} / Python {@code tritonserver.DataType}.
 * {@link #byteSize()} and {@link #typeString()} use the documented C values so they
 * work without loading native libraries (needed for unit tests on non-Linux hosts).
 */
public enum TritonDataType {
    INVALID(0, 0, "INVALID"),
    BOOL(1, 1, "BOOL"),
    UINT8(2, 1, "UINT8"),
    UINT16(3, 2, "UINT16"),
    UINT32(4, 4, "UINT32"),
    UINT64(5, 8, "UINT64"),
    INT8(6, 1, "INT8"),
    INT16(7, 2, "INT16"),
    INT32(8, 4, "INT32"),
    INT64(9, 8, "INT64"),
    FP16(10, 2, "FP16"),
    FP32(11, 4, "FP32"),
    FP64(12, 8, "FP64"),
    BYTES(13, 0, "BYTES"),
    BF16(14, 2, "BF16");

    private final int code;
    private final int byteSize;
    private final String typeString;

    TritonDataType(int code, int byteSize, String typeString) {
        this.code = code;
        this.byteSize = byteSize;
        this.typeString = typeString;
    }

    /** Native {@code TRITONSERVER_DataType} ordinal. */
    public int code() {
        return code;
    }

    /**
     * Element size in bytes for fixed-width types.
     *
     * <p>For {@link #BYTES} and {@link #INVALID} returns 0 (same as
     * {@code TRITONSERVER_DataTypeByteSize}).
     */
    public int byteSize() {
        return byteSize;
    }

    /** Triton wire name (e.g. {@code "FP32"}, {@code "INT32"}). */
    public String typeString() {
        return typeString;
    }

    public static TritonDataType fromCode(int code) {
        for (TritonDataType t : values()) {
            if (t.code == code) {
                return t;
            }
        }
        throw new TritonInvalidArgumentException("Unknown DataType code: " + code);
    }

    /**
     * Parse Triton datatype string (e.g. {@code "FP32"}).
     *
     * <p>Matching is case-sensitive to the C API string form.
     */
    public static TritonDataType fromString(String name) {
        if (name == null || name.isEmpty()) {
            throw new TritonInvalidArgumentException("DataType string must be non-empty");
        }
        for (TritonDataType t : values()) {
            if (t.typeString.equals(name)) {
                return t;
            }
        }
        throw new TritonInvalidArgumentException("Unknown DataType string: " + name);
    }
}
