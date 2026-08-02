package org.bytedeco.pytorch.serving.tensorrt.enums;

import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtInvalidArgumentException;

/**
 * TensorRT tensor data type.
 *
 * <p>Codes match {@code nvinfer1::DataType} / Python {@code tensorrt.DataType}
 * (bytedeco {@code org.bytedeco.tensorrt.global.nvinfer.DataType}).
 *
 * <p>{@link #byteSize()} is documented element size so unit tests work without
 * loading native libraries.
 */
public enum TRTDataType {
    /** 32-bit floating point. Python: {@code tensorrt.float32} / {@code DataType.FLOAT}. */
    FLOAT(0, 4, "FLOAT"),
    /** 16-bit floating point. Python: {@code tensorrt.float16} / {@code DataType.HALF}. */
    HALF(1, 2, "HALF"),
    /** 8-bit signed integer. */
    INT8(2, 1, "INT8"),
    /** 32-bit signed integer. */
    INT32(3, 4, "INT32"),
    /** Boolean. */
    BOOL(4, 1, "BOOL"),
    /** 8-bit unsigned integer. */
    UINT8(5, 1, "UINT8"),
    /** 8-bit floating point (E4M3). */
    FP8(6, 1, "FP8"),
    /** Brain float 16. */
    BF16(7, 2, "BF16"),
    /** 64-bit signed integer. */
    INT64(8, 8, "INT64"),
    /** 4-bit signed integer (packed; byteSize is nominal). */
    INT4(9, 0, "INT4"),
    /** 4-bit floating point (packed; byteSize is nominal). */
    FP4(10, 0, "FP4"),
    /** Exponent-only 8-bit float used for quantization scales (TRT 10+). */
    E8M0(11, 1, "E8M0");

    private final int code;
    private final int byteSize;
    private final String typeString;

    TRTDataType(int code, int byteSize, String typeString) {
        this.code = code;
        this.byteSize = byteSize;
        this.typeString = typeString;
    }

    /** Native {@code nvinfer1::DataType} ordinal. */
    public int code() {
        return code;
    }

    /**
     * Element size in bytes for fixed-width types.
     *
     * <p>Returns 0 for packed types ({@link #INT4}, {@link #FP4}) where size is
     * not a simple per-element byte count.
     */
    public int byteSize() {
        return byteSize;
    }

    /** TensorRT type name (e.g. {@code "FLOAT"}, {@code "HALF"}). */
    public String typeString() {
        return typeString;
    }

    public static TRTDataType fromCode(int code) {
        for (TRTDataType t : values()) {
            if (t.code == code) {
                return t;
            }
        }
        throw new TrtInvalidArgumentException("Unknown DataType code: " + code);
    }

    /**
     * Parse type string (case-insensitive). Accepts TensorRT names and common
     * aliases used in Python ({@code float32}, {@code float16}, {@code int32}, …).
     */
    public static TRTDataType fromString(String name) {
        if (name == null || name.isEmpty()) {
            throw new TrtInvalidArgumentException("DataType string must be non-empty");
        }
        String n = name.trim();
        for (TRTDataType t : values()) {
            if (t.typeString.equalsIgnoreCase(n)) {
                return t;
            }
        }
        return switch (n.toLowerCase()) {
            case "float", "float32", "fp32" -> FLOAT;
            case "float16", "fp16", "half" -> HALF;
            case "bfloat16", "bf16" -> BF16;
            case "int8" -> INT8;
            case "int32" -> INT32;
            case "int64" -> INT64;
            case "bool", "boolean" -> BOOL;
            case "uint8", "byte" -> UINT8;
            case "fp8" -> FP8;
            case "int4" -> INT4;
            case "fp4" -> FP4;
            default -> throw new TrtInvalidArgumentException("Unknown DataType string: " + name);
        };
    }
}
