package org.bytedeco.pytorch.dataframe.enums;

/** Safetensors-aligned tensor element types. */
public enum TensorDType {
    F64("F64", 8), F32("F32", 4), F16("F16", 2), BF16("BF16", 2),
    I64("I64", 8), I32("I32", 4), I16("I16", 2), I8("I8", 1), U8("U8", 1),
    Q4("Q4", 1), Q8("Q8", 1), BOOL("BOOL", 1);

    private final String name;
    private final int bytesPerElement;

    TensorDType(String name, int bytesPerElement) {
        this.name = name;
        this.bytesPerElement = bytesPerElement;
    }

    public String getName() { return name; }
    public int size() { return bytesPerElement; }

    public static TensorDType fromName(String n) {
        for (TensorDType t : values()) if (t.name.equalsIgnoreCase(n)) return t;
        throw new IllegalArgumentException("Unknown TensorDType: " + n);
    }

    /** Alias used by migrated TensorData loaders. */
    public static TensorDType fromString(String n) {
        return fromName(n);
    }
}
