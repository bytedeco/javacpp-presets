package org.bytedeco.pytorch.data.safetensors;

import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.Locale;

/**
 * Dtypes used by the safetensors format (little-endian on disk).
 */
public enum SafeDType {
    F64("F64", 8, ScalarType.Double),
    F32("F32", 4, ScalarType.Float),
    F16("F16", 2, ScalarType.Half),
    BF16("BF16", 2, ScalarType.BFloat16),
    I64("I64", 8, ScalarType.Long),
    I32("I32", 4, ScalarType.Int),
    I16("I16", 2, ScalarType.Short),
    I8("I8", 1, ScalarType.Char),
    U8("U8", 1, ScalarType.Byte),
    BOOL("BOOL", 1, ScalarType.Bool);

    private final String name;
    private final int bytes;
    private final ScalarType torch;

    SafeDType(String name, int bytes, ScalarType torch) {
        this.name = name;
        this.bytes = bytes;
        this.torch = torch;
    }

    public String typeName() { return name; }
    public int sizeBytes() { return bytes; }
    public ScalarType toTorch() { return torch; }

    /**
     * Whether on-disk little-endian layout matches torch storage so
     * {@code from_blob} can share the mapping without conversion.
     * F16/BF16 are native; BOOL is not (torch may pack differently).
     */
    public boolean isNativeLayout() {
        return this != BOOL;
    }

    public static SafeDType fromString(String s) {
        if (s == null) return null;
        switch (s.toUpperCase(Locale.ROOT)) {
            case "F64": case "FLOAT64": case "DOUBLE": return F64;
            case "F32": case "FLOAT32": case "FLOAT": return F32;
            case "F16": case "FLOAT16": case "HALF": return F16;
            case "BF16": case "BFLOAT16": return BF16;
            case "I64": case "INT64": case "LONG": return I64;
            case "I32": case "INT32": case "INT": return I32;
            case "I16": case "INT16": case "SHORT": return I16;
            case "I8": case "INT8": return I8;
            case "U8": case "UINT8": return U8;
            case "BOOL": case "BOOLEAN": return BOOL;
            default: return null;
        }
    }

    /**
     * Map a torch ScalarType to safetensors dtype.
     *
     * <p><b>JavaCPP pitfall:</b> {@code Tensor.scalar_type()} often returns a
     * non-canonical enum proxy ({@code name=null}, {@code ordinal=0}) whose
     * {@code switch} identity matches {@link ScalarType#Byte}. Always
     * {@link ScalarType#intern()} first so case labels resolve by real value
     * ({@code Float.value=6}, etc.). Matching on {@code st.value} is an
     * equivalent alternative.
     */
    public static SafeDType fromTorch(ScalarType st) {
        if (st == null) return F32;
        // intern() maps the native-backed proxy onto the canonical enum constant
        ScalarType s = st.intern();
        switch (s) {
            case Double: return F64;
            case Float: return F32;
            case Half: return F16;
            case BFloat16: return BF16;
            case Long: return I64;
            case Int: return I32;
            case Short: return I16;
            case Char: return I8;
            case Byte: return U8;
            case Bool: return BOOL;
            default: return F32;
        }
    }
}
