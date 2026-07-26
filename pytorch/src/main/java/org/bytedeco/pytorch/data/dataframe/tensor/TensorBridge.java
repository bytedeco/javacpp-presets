package org.bytedeco.pytorch.data.dataframe.tensor;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.c10.LongHeaderOnlyArrayRef;
import org.bytedeco.pytorch.data.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.data.dataframe.dtype.TensorData;
import org.bytedeco.pytorch.data.dataframe.dtype.VectorData;
import org.bytedeco.pytorch.data.dataframe.enums.TensorDType;
import org.bytedeco.pytorch.data.numpy.DType;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Central multi-dimensional bridge among javacpp-pytorch {@link Tensor},
 * {@link TensorData}, {@link NDArray}, {@link VectorData}, and {@link EmbeddingData}.
 *
 * <p>Explicitly supports ranks 0–4 (scalar … NCHW-style). Higher ranks are allowed
 * and treated as flatten-friendly bulk tensors.
 *
 * <p>Zero extra dependencies — pure Java + existing JavaCPP torch bindings.
 * Copy-based by default; optional native attach on {@link TensorData} for zero-copy
 * when the caller already holds a live Tensor.
 */
public final class TensorBridge {
    private TensorBridge() {}

    // ---- shape helpers ------------------------------------------------------

    public static long[] sizesOf(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        return sizesAsArray(t.sizes());
    }

    public static int[] intShapeOf(Tensor t) {
        long[] s = sizesOf(t);
        int[] out = new int[s.length];
        for (int i = 0; i < s.length; i++) {
            if (s[i] > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("dim " + i + " exceeds int range: " + s[i]);
            }
            out[i] = (int) s[i];
        }
        return out;
    }

    public static long[] toLongShape(int[] shape) {
        if (shape == null) return new long[0];
        long[] out = new long[shape.length];
        for (int i = 0; i < shape.length; i++) out[i] = shape[i];
        return out;
    }

    public static int[] toIntShape(long[] shape) {
        if (shape == null) return new int[0];
        int[] out = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("dim " + i + " exceeds int range: " + shape[i]);
            }
            out[i] = (int) shape[i];
        }
        return out;
    }

    public static int rank(Tensor t) {
        return sizesOf(t).length;
    }

    public static long numel(int[] shape) {
        long n = 1;
        if (shape != null) for (int s : shape) n *= s;
        return n;
    }

    // ---- dtype maps ---------------------------------------------------------

    public static TensorDType toTensorDType(ScalarType st) {
        if (st == null) return TensorDType.F32;
        // JavaCPP: Tensor.scalar_type() returns a non-canonical proxy — intern first
        // or switch falls through to Byte (ordinal 0).
        return switch (st.intern()) {
            case Double -> TensorDType.F64;
            case Float -> TensorDType.F32;
            case Half -> TensorDType.F16;
            case BFloat16 -> TensorDType.BF16;
            case Long -> TensorDType.I64;
            case Int -> TensorDType.I32;
            case Short -> TensorDType.I16;
            case Char -> TensorDType.I8;
            case Byte -> TensorDType.U8;
            case Bool -> TensorDType.BOOL;
            default -> TensorDType.F32;
        };
    }

    public static ScalarType toScalarType(TensorDType dt) {
        if (dt == null) return ScalarType.Float;
        return switch (dt) {
            case F64 -> ScalarType.Double;
            case F32 -> ScalarType.Float;
            case F16 -> ScalarType.Half;
            case BF16 -> ScalarType.BFloat16;
            case I64 -> ScalarType.Long;
            case I32 -> ScalarType.Int;
            case I16 -> ScalarType.Short;
            case I8 -> ScalarType.Char;
            case U8 -> ScalarType.Byte;
            case BOOL -> ScalarType.Bool;
            case Q4, Q8 -> ScalarType.Float; // dequant path not handled here
        };
    }

    public static DType toNumpyDType(TensorDType dt) {
        if (dt == null) return DType.FLOAT32;
        return switch (dt) {
            case F64 -> DType.FLOAT64;
            case F32, F16, BF16, Q4, Q8 -> DType.FLOAT32;
            case I64 -> DType.INT64;
            case I32 -> DType.INT32;
            case I16 -> DType.INT16;
            case I8 -> DType.INT8;
            case U8 -> DType.UINT8;
            case BOOL -> DType.BOOL;
        };
    }

    public static TensorDType fromNumpyDType(DType d) {
        if (d == null) return TensorDType.F32;
        return switch (d) {
            case FLOAT64 -> TensorDType.F64;
            case FLOAT32 -> TensorDType.F32;
            case FLOAT16 -> TensorDType.F16;
            case INT64 -> TensorDType.I64;
            case INT32 -> TensorDType.I32;
            case INT16 -> TensorDType.I16;
            case INT8 -> TensorDType.I8;
            case UINT8 -> TensorDType.U8;
            case BOOL -> TensorDType.BOOL;
            default -> TensorDType.F32;
        };
    }

    public static ScalarType toScalarType(DType d) {
        return toScalarType(fromNumpyDType(d));
    }

    // ---- Tensor → TensorData ------------------------------------------------

    /**
     * Copy a torch Tensor into a {@link TensorData} (CPU, contiguous).
     * Floating types land as float[] (F32 storage) or F64-labelled float promotion;
     * integer types are promoted to float storage for TensorData compatibility,
     * with dtype metadata preserved when possible.
     */
    public static TensorData toTensorData(Tensor t) {
        return toTensorData(t, false);
    }

    /**
     * @param attachNative when true, also {@link TensorData#attachNativeTensor(Tensor)}
     *                     the contiguous CPU tensor (caller must keep it alive).
     */
    public static TensorData toTensorData(Tensor t, boolean attachNative) {
        Objects.requireNonNull(t, "tensor");
        Tensor cpu = t.contiguous().cpu();
        int[] shape = intShapeOf(cpu);
        ScalarType st = cpu.scalar_type();
        TensorDType td = toTensorDType(st);
        float[] data = extractFloatData(cpu, st);
        TensorData out = new TensorData(data, shape);
        // preserve logical dtype when not F32
        if (td != TensorDType.F32) {
            // TensorData(float[], shape) forces F32; rebuild via buffer path for metadata
            out = TensorData.fromFloatData(data, shape, td);
        }
        if (attachNative) {
            out.attachNativeTensor(cpu);
        }
        return out;
    }

    // ---- TensorData → Tensor ------------------------------------------------

    public static Tensor toTensor(TensorData td) {
        Objects.requireNonNull(td, "tensorData");
        if (td.hasNativeTensor()) {
            Tensor n = td.getNativeTensor();
            if (n != null && !n.isNull()) return n;
        }
        float[] data = td.getData();
        long[] shape = toLongShape(td.getShape());
        Tensor t = torch.tensor(data);
        if (shape.length > 0) t = t.reshape(shape);
        ScalarType want = toScalarType(td.getDType());
        if (want != ScalarType.Float) {
            t = t.to(want);
        }
        return t;
    }

    // ---- NDArray ↔ Tensor (delegate to NP which already handles all dtypes) ---

    public static Tensor toTensor(NDArray arr) {
        Objects.requireNonNull(arr, "ndarray");
        return org.bytedeco.pytorch.data.numpy.NP.toTensor(arr);
    }

    public static NDArray toNDArray(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        return org.bytedeco.pytorch.data.numpy.NP.fromTensor(t);
    }

    public static TensorData toTensorData(NDArray arr) {
        Objects.requireNonNull(arr, "ndarray");
        int[] shape = toIntShape(arr.shape);
        TensorDType td = fromNumpyDType(arr.dtype);
        if (NDArray.isFloatFamily(arr.dtype)) {
            float[] f = arr.asFloatArray();
            return TensorData.fromFloatData(f, shape, td);
        }
        float[] f = new float[(int) arr.size];
        for (int i = 0; i < f.length; i++) f[i] = (float) arr.getLong(i);
        return TensorData.fromFloatData(f, shape, td);
    }

    public static NDArray toNDArray(TensorData td) {
        Objects.requireNonNull(td, "tensorData");
        float[] data = td.getData();
        long[] shape = toLongShape(td.getShape());
        DType dt = toNumpyDType(td.getDType());
        if (dt == DType.FLOAT64) {
            double[] d = new double[data.length];
            for (int i = 0; i < data.length; i++) d[i] = data[i];
            return new NDArray(d, shape);
        }
        return new NDArray(data, shape);
    }

    // ---- VectorData / EmbeddingData -----------------------------------------

    public static Tensor toTensor(VectorData vd) {
        Objects.requireNonNull(vd, "vectorData");
        int[] shape = vd.getShape();
        if (shape == null || shape.length == 0) shape = new int[]{vd.getVectorSize()};
        long[] lshape = toLongShape(shape);
        String vt = vd.getVectorType();
        if ("float32".equals(vt)) {
            float[] f = vd.getFloatVector();
            Tensor t = torch.tensor(f);
            return lshape.length > 0 ? t.reshape(lshape) : t;
        }
        if ("int32".equals(vt)) {
            int[] iv = vd.getIntVector();
            Tensor t = torch.tensor(iv);
            return lshape.length > 0 ? t.reshape(lshape) : t;
        }
        double[] d = vd.getAsDoubleArray();
        Tensor t = torch.tensor(d);
        return lshape.length > 0 ? t.reshape(lshape) : t;
    }

    public static VectorData toVectorData(Tensor t) {
        return toVectorData(t, "from_tensor");
    }

    public static VectorData toVectorData(Tensor t, String name) {
        Objects.requireNonNull(t, "tensor");
        Tensor cpu = t.contiguous().cpu();
        int[] shape = intShapeOf(cpu);
        long n = cpu.numel();
        ScalarType st = cpu.scalar_type();
        if (st == ScalarType.Float || st == ScalarType.Half || st == ScalarType.BFloat16) {
            Tensor f = st == ScalarType.Float ? cpu : cpu.to(ScalarType.Float);
            FloatPointer ptr = f.data_ptr_float();
            float[] data = new float[(int) n];
            for (int i = 0; i < n; i++) data[i] = ptr.get(i);
            if (shape.length <= 1) return new VectorData(data, name);
            double[] d = new double[data.length];
            for (int i = 0; i < data.length; i++) d[i] = data[i];
            return new VectorData(d, shape, name);
        }
        if (st == ScalarType.Int || st == ScalarType.Short || st == ScalarType.Char || st == ScalarType.Byte) {
            Tensor i32 = cpu.to(ScalarType.Int);
            IntPointer ptr = i32.data_ptr_int();
            int[] data = new int[(int) n];
            for (int i = 0; i < n; i++) data[i] = ptr.get(i);
            if (shape.length <= 1) return new VectorData(data, name);
            double[] d = new double[data.length];
            for (int i = 0; i < data.length; i++) d[i] = data[i];
            return new VectorData(d, shape, name);
        }
        Tensor f64 = cpu.to(ScalarType.Double);
        DoublePointer ptr = f64.data_ptr_double();
        double[] data = new double[(int) n];
        for (int i = 0; i < n; i++) data[i] = ptr.get(i);
        if (shape.length <= 1) return new VectorData(data, name);
        return new VectorData(data, shape, name);
    }

    public static Tensor toTensor(EmbeddingData emb) {
        Objects.requireNonNull(emb, "embedding");
        return torch.tensor(emb.getVector());
    }

    public static EmbeddingData toEmbeddingData(Tensor t, String modelName) {
        Objects.requireNonNull(t, "tensor");
        Tensor cpu = t.contiguous().cpu().to(ScalarType.Float).reshape(new long[]{-1});
        long n = cpu.numel();
        FloatPointer ptr = cpu.data_ptr_float();
        float[] data = new float[(int) n];
        for (int i = 0; i < n; i++) data[i] = ptr.get(i);
        return new EmbeddingData(data, modelName == null ? "tensor" : modelName);
    }

    // ---- generic cell → float[] (1-D) for ANN / vector stores ---------------

    /**
     * Extract a 1-D float vector from common cell types.
     * Multi-dim {@link TensorData} / shaped {@link VectorData} are <em>flattened</em>
     * (useful for ANN helpers); vector-store upsert should only accept already-1D
     * EMBEDDING/VECTOR cells per product policy.
     */
    public static float[] asFloatVector(Object cell) {
        if (cell == null) return null;
        if (cell instanceof float[] f) return f;
        if (cell instanceof double[] d) {
            float[] out = new float[d.length];
            for (int i = 0; i < d.length; i++) out[i] = (float) d[i];
            return out;
        }
        if (cell instanceof EmbeddingData e) return e.getVector();
        if (cell instanceof VectorData vd) {
            if ("float32".equals(vd.getVectorType())) {
                float[] f = vd.getFloatVector();
                return f;
            }
            double[] d = vd.getAsDoubleArray();
            float[] out = new float[d.length];
            for (int i = 0; i < d.length; i++) out[i] = (float) d[i];
            return out;
        }
        if (cell instanceof TensorData td) return td.getData();
        if (cell instanceof Tensor t) {
            return toEmbeddingData(t, "tmp").getVector();
        }
        if (cell instanceof NDArray arr) return arr.asFloatArray();
        if (cell instanceof Number n) return new float[]{n.floatValue()};
        return null;
    }

    /** True when cell is a 1-D dense vector suitable for vector-DB indexing. */
    public static boolean isIndexableVectorCell(Object cell) {
        if (cell == null) return false;
        if (cell instanceof float[] f) return f.length > 0;
        if (cell instanceof double[] d) return d.length > 0;
        if (cell instanceof EmbeddingData) return true;
        if (cell instanceof VectorData vd) {
            int[] sh = vd.getShape();
            return sh == null || sh.length <= 1;
        }
        // multi-dim TensorData is NOT indexable per product policy
        return false;
    }

    // ---- equality helper for benchmarks -------------------------------------

    public static boolean shapesEqual(int[] a, int[] b) {
        return Arrays.equals(a, b);
    }

    public static boolean shapesEqual(long[] a, int[] b) {
        if (a == null || b == null) return a == null && b == null;
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static boolean approxEqual(float[] a, float[] b, float eps) {
        if (a == null || b == null) return a == b;
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (Float.isNaN(a[i]) && Float.isNaN(b[i])) continue;
            if (Math.abs(a[i] - b[i]) > eps) return false;
        }
        return true;
    }

    // ---- internals ----------------------------------------------------------

    private static float[] extractFloatData(Tensor cpu, ScalarType st) {
        long n = cpu.numel();
        float[] data = new float[(int) n];
        if (st == ScalarType.Float) {
            FloatPointer ptr = cpu.data_ptr_float();
            for (int i = 0; i < n; i++) data[i] = ptr.get(i);
            return data;
        }
        if (st == ScalarType.Double) {
            DoublePointer ptr = cpu.data_ptr_double();
            for (int i = 0; i < n; i++) data[i] = (float) ptr.get(i);
            return data;
        }
        // promote everything else via float
        Tensor f = cpu.to(ScalarType.Float);
        FloatPointer ptr = f.data_ptr_float();
        for (int i = 0; i < n; i++) data[i] = ptr.get(i);
        return data;
    }

    private static long[] sizesAsArray(LongHeaderOnlyArrayRef ref) {
        long len = ref.size();
        if (len == 0) return new long[0];
        return ref.vec().get();
    }
}
