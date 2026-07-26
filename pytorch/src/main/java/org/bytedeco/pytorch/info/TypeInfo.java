package org.bytedeco.pytorch.info;
import org.bytedeco.pytorch.c10.*;

import org.bytedeco.pytorch.global.torch.ScalarType;

public abstract class TypeInfo {
    public final String type;
    public final int bits;

    protected TypeInfo(String type, int bits) {
        this.type = type;
        this.bits = bits;
    }

    public static IInfo iinfo2(ScalarType type) {
        switch (type) {
            case Byte:   return new IInfo("int8", 8, -128, 127);
            case QUInt8: return new IInfo("uint8", 8, 0, 255);
            case Short:  return new IInfo("int16", 16, -32768, 32767);
            case UInt16: return new IInfo("uint16", 16, 0, 65535);
            case Int:    return new IInfo("int32", 32, Integer.MIN_VALUE, Integer.MAX_VALUE);
            case UInt32: return new IInfo("uint32", 32, 0, 4294967295L);
            case Long:   return new IInfo("int64", 64, Long.MIN_VALUE, Long.MAX_VALUE);
            case Bool:   return new IInfo("bool", 1, 0, 1);
            default:
                throw new IllegalArgumentException(type + " is not an integer type.");
        }
    }

    public static IInfo iinfo(ScalarType type) {
        switch (type) {
            case Byte:
                // PyTorch 中 kByte 是 uint8
                return new IInfo("uint8", 8, 0, 255);

            case Char:
                // PyTorch 中 kChar 是 int8
                return new IInfo("int8", 8, -128, 127);

            case Short:
                return new IInfo("int16", 16, -32768, 32767);

            case Int:
                return new IInfo("int32", 32, Integer.MIN_VALUE, Integer.MAX_VALUE);

            case Long:
                return new IInfo("int64", 64, Long.MIN_VALUE, Long.MAX_VALUE);

            case UInt16:
                return new IInfo("uint16", 16, 0, 65535);

            case UInt32:
                return new IInfo("uint32", 32, 0, 4294967295L);

            case Bool:
                return new IInfo("bool", 1, 0, 1);

            case UInt64: return new IInfo("uint64", 64, 0, -1); // 注意：Java long 无法直接表示 uint64_max，通常设为 -1 (all bits 1)

            // --- 量化整数 (QInt) ---
            case QInt8:  return new IInfo("qint8", 8, -128, 127);
            case QUInt8: return new IInfo("quint8", 8, 0, 255);
            case QInt32: return new IInfo("qint32", 32, Integer.MIN_VALUE, Integer.MAX_VALUE);
            case QUInt4x2: return new IInfo("quint4x2", 4, 0, 15); // 每个元素 4bit
            case QUInt2x4: return new IInfo("quint2x4", 2, 0, 3);  // 每个元素 2bit

            // --- 位存储类型 (Bits) ---
            case Bits8:  return new IInfo("bits8", 8, 0, 255);
            case Bits16: return new IInfo("bits16", 16, 0, 65535);

            // --- 窄位宽整数 (N-bit) ---
            case Int4:   return new IInfo("int4", 4, -8, 7);
            case UInt4:  return new IInfo("uint4", 4, 0, 15);
            case Int2:   return new IInfo("int2", 2, -2, 1);
            case UInt2:  return new IInfo("uint2", 2, 0, 3);
            case Int1:   return new IInfo("int1", 1, -1, 0);
            case UInt1:  return new IInfo("uint1", 1, 0, 1);
            default:
                throw new IllegalArgumentException("Type " + type + " is not a supported integer type.");
        }
    }

    public static FInfo finfo(ScalarType type) {
        switch (type) {
            case Float: // float32
                return new FInfo("float32", 32, -3.4028235e+38, 3.4028235e+38, 1.1920929e-07, 1.1754944e-38, 7);

            case Double: // float64
                return new FInfo("float64", 64, -1.7976931348623157e+308, 1.7976931348623157e+308, 2.220446049250313e-16, 2.2250738585072014e-308, 15);

            case Half: // float16 (IEEE 754)
                return new FInfo("float16", 16, -65504.0, 65504.0, 0.0009765625, 6.103515625e-05, 3);

            case BFloat16: // bfloat16 (Brain Float)
                return new FInfo("bfloat16", 16, -3.389531389251535e+38, 3.389531389251535e+38, 0.0078125, 1.1754943508222875e-38, 2);

            case Float8_e4m3fn:
                return new FInfo("float8_e4m3fn", 8, -448.0, 448.0, 0.125, 0.015625, 1);

            case Float8_e5m2:
                return new FInfo("float8_e5m2", 8, -57344.0, 57344.0, 0.25, 0.000015258789, 1);

            default:
                throw new IllegalArgumentException(type + " is not a floating point type.");
        }
    }
}



    /**
     * 实现 torch.finfo(dtype)
     */
//    public static FInfo finfo(ScalarType dtype) {
//        if (dtype.equals(kDouble()) || dtype.equals(kFloat())) {
//            return new FInfo(1.1920929e-07, 3.4028235e+38, -3.4028235e+38, 1.1754944e-38, 32);
//        } else if (dtype.equals(kFloat()) || dtype.equals(kDouble())) {
//            return new FInfo(2.220446049250313e-16, 1.7976931348623157e+308, -1.7976931348623157e+308, 2.2250738585072014e-308, 64);
//        } else if (dtype.equals(kBFloat16()) || dtype.equals(kHalf())) {
//            return new FInfo(0.00097656, 65504.0, -65504.0, 6.1035e-05, 16);
//        }
//        throw new IllegalArgumentException("Only floating point dtypes are supported by finfo");
//    }
//
//    /**
//     * 实现 torch.iinfo(dtype)
//     */
//    public static IInfo iinfo(ScalarType dtype) {
//        if (dtype.equals(kInt32()) || dtype.equals(kInt())) {
//            return new IInfo(Integer.MAX_VALUE, Integer.MIN_VALUE, 32);
//        } else if (dtype.equals(kInt64()) || dtype.equals(kLong())) {
//            return new IInfo(Long.MAX_VALUE, Long.MIN_VALUE, 64);
//        } else if (dtype.equals(kQInt8())) {
//            return new IInfo(Byte.MAX_VALUE, Byte.MIN_VALUE, 8);
//        }
//        throw new IllegalArgumentException("Only integer dtypes are supported by iinfo");
//    }
//}
