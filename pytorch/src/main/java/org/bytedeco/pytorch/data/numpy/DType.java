package org.bytedeco.pytorch.data.numpy;

import org.bytedeco.pytorch.global.torch.ScalarType;

/**
 * NumPy dtype descriptors used by {@code .npy}/{@code .npz} I/O and {@link NDArray}.
 * Little-endian only (standard for modern NumPy files).
 */
public enum DType {
    FLOAT64("<f8", 8, ScalarType.Double, false),
    FLOAT32("<f4", 4, ScalarType.Float, false),
    FLOAT16("<f2", 2, ScalarType.Half, false),
    INT64("<i8", 8, ScalarType.Long, false),
    INT32("<i4", 4, ScalarType.Int, false),
    INT16("<i2", 2, ScalarType.Short, false),
    INT8("|i1", 1, ScalarType.Char, false),
    UINT8("|u1", 1, ScalarType.Byte, false),
    BOOL("|b1", 1, ScalarType.Bool, false),
    /** Interleaved float32 complex — stored as double pairs in {@link NDArray}. */
    COMPLEX64("<c8", 8, ScalarType.ComplexFloat, true),
    /** Interleaved float64 complex. */
    COMPLEX128("<c16", 16, ScalarType.ComplexDouble, true);

    private final String descriptor;
    private final int byteSize;
    private final ScalarType torchType;
    private final boolean complex;

    DType(String descriptor, int byteSize, ScalarType torchType, boolean complex) {
        this.descriptor = descriptor;
        this.byteSize = byteSize;
        this.torchType = torchType;
        this.complex = complex;
    }

    public String getDescriptor() { return descriptor; }
    public int getByteSize() { return byteSize; }
    public ScalarType toTorch() { return torchType; }
    public boolean isComplex() { return complex; }

    public static DType fromDescriptor(String desc) {
        if (desc == null) return FLOAT64;
        String d = desc.trim();
        for (DType t : values()) {
            if (t.descriptor.equals(d)) return t;
        }
        if (d.endsWith("f8") || d.equals("float64") || d.equals("f8")) return FLOAT64;
        if (d.endsWith("f4") || d.equals("float32") || d.equals("f4")) return FLOAT32;
        if (d.endsWith("f2") || d.equals("float16") || d.equals("f2")) return FLOAT16;
        if (d.endsWith("i8") || d.equals("int64") || d.equals("i8")) return INT64;
        if (d.endsWith("i4") || d.equals("int32") || d.equals("i4")) return INT32;
        if (d.endsWith("i2") || d.equals("int16") || d.equals("i2")) return INT16;
        if (d.endsWith("i1") || d.equals("int8") || d.equals("i1")) return INT8;
        if (d.endsWith("u1") || d.equals("uint8") || d.equals("u1")) return UINT8;
        if (d.endsWith("b1") || d.equals("bool") || d.equals("b1")) return BOOL;
        if (d.endsWith("c16") || d.equals("complex128") || d.equals("c16")) return COMPLEX128;
        if (d.endsWith("c8") || d.equals("complex64") || d.equals("c8")) return COMPLEX64;
        return FLOAT64;
    }

    public static DType fromTorch(ScalarType st) {
        if (st == null) return FLOAT32;
        // JavaCPP: Tensor.scalar_type() returns a non-canonical proxy — intern first
        // or switch falls through to Byte (ordinal 0).
        ScalarType s = st.intern();
        switch (s) {
            case Double: return FLOAT64;
            case Float: return FLOAT32;
            case Half: return FLOAT16;
            case Long: return INT64;
            case Int: return INT32;
            case Short: return INT16;
            case Char: return INT8;
            case Byte: return UINT8;
            case Bool: return BOOL;
            case ComplexDouble: return COMPLEX128;
            case ComplexFloat: return COMPLEX64;
            default: return FLOAT32;
        }
    }
}
