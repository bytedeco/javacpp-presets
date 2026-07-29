package org.bytedeco.pytorch.utils.vista;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;

/**
 * Tensor extraction / shape formatting helpers used while building the
 * forward-pass graph.
 *
 * <p>Analogous to torchvista {@code tensor_utils.extract_tensors_from_obj} and
 * {@code engine.format_dims}. JavaCPP cannot hang arbitrary attributes on a
 * C++ {@link Tensor}, so tensor→source tagging is done via a
 * {@code Map<Long,String>} keyed by {@link #tensorKey(Tensor)} (native address).
 */
public final class TensorUtils {
    private TensorUtils() {}

    /**
     * Stable key for a live Tensor during a single trace. Prefer
     * {@link Tensor#address()} (native pointer); fall back to identity hash.
     */
    public static long tensorKey(Tensor t) {
        if (t == null || t.isNull()) return 0L;
        long addr = t.address();
        return addr != 0L ? addr : System.identityHashCode(t);
    }

    /** Format dims like torchvista: {@code (2, 10)} or {@code ( )} for scalar. */
    public static String formatDims(long[] shape) {
        if (shape == null || shape.length == 0) {
            return "( )";
        }
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        sb.append(')');
        return sb.toString();
    }

    public static String formatDims(Tensor t) {
        if (t == null || t.isNull()) return "( )";
        try {
            return formatDims(t.shape());
        } catch (Throwable e) {
            return "(?)";
        }
    }

    public static long[] safeShape(Tensor t) {
        if (t == null || t.isNull()) return new long[0];
        try {
            long[] s = t.shape();
            return s == null ? new long[0] : s;
        } catch (Throwable e) {
            return new long[0];
        }
    }

    public static String safeDtype(Tensor t) {
        if (t == null || t.isNull()) return "unknown";
        try {
            return String.valueOf(t.scalar_type());
        } catch (Throwable e) {
            return "unknown";
        }
    }

    /**
     * Walk an arbitrary object tree and collect live Tensors (depth-first).
     * Supports: Tensor, Tensor[], Collection, Map, TensorVector, Object[].
     */
    public static List<Tensor> extractTensors(Object obj) {
        List<Tensor> out = new ArrayList<>();
        extractInto(obj, out);
        return out;
    }

    private static void extractInto(Object obj, List<Tensor> out) {
        if (obj == null) return;
        if (obj instanceof Tensor) {
            Tensor t = (Tensor) obj;
            if (!t.isNull()) out.add(t);
            return;
        }
        if (obj instanceof TensorVector) {
            TensorVector v = (TensorVector) obj;
            try {
                long n = v.size();
                for (long i = 0; i < n; i++) {
                    Tensor t = v.get(i);
                    if (t != null && !t.isNull()) out.add(t);
                }
            } catch (Throwable ignored) {}
            return;
        }
        if (obj instanceof Tensor[]) {
            for (Tensor t : (Tensor[]) obj) {
                if (t != null && !t.isNull()) out.add(t);
            }
            return;
        }
        if (obj instanceof Object[]) {
            for (Object o : (Object[]) obj) extractInto(o, out);
            return;
        }
        if (obj instanceof Collection) {
            for (Object o : (Collection<?>) obj) extractInto(o, out);
            return;
        }
        if (obj instanceof Map) {
            for (Object o : ((Map<?, ?>) obj).values()) extractInto(o, out);
            return;
        }
        // POJO results (VectorQuantizer.Result, etc.): public Tensor fields
        if (!(obj instanceof Number) && !(obj instanceof CharSequence) && !(obj instanceof Boolean)) {
            try {
                for (java.lang.reflect.Field f : obj.getClass().getFields()) {
                    if (Tensor.class.isAssignableFrom(f.getType())) {
                        Object v = f.get(obj);
                        if (v instanceof Tensor) {
                            Tensor t = (Tensor) v;
                            if (t != null && !t.isNull()) out.add(t);
                        }
                    }
                }
            } catch (Throwable ignored) {}
        }
        // bare Number / String / boolean — not tensors
    }

    /**
     * Format a single argument for func_info (positional/keyword args popup).
     * Matches torchvista {@code format_arg}: tensors become
     * {@code {_type:tensor, shape, dtype}}.
     */
    public static Object formatArg(Object value) {
        if (value == null) return null;
        if (value instanceof Tensor) {
            Tensor t = (Tensor) value;
            Map<String, Object> m = new java.util.LinkedHashMap<>();
            m.put("_type", "tensor");
            long[] shape = safeShape(t);
            List<Long> shapeList = new ArrayList<>(shape.length);
            for (long s : shape) shapeList.add(s);
            m.put("shape", shapeList);
            m.put("dtype", safeDtype(t));
            return m;
        }
        if (value instanceof Number || value instanceof Boolean || value instanceof String) {
            return value;
        }
        if (value instanceof Collection) {
            List<Object> list = new ArrayList<>();
            for (Object o : (Collection<?>) value) list.add(formatArg(o));
            return list;
        }
        if (value instanceof Object[]) {
            List<Object> list = new ArrayList<>();
            for (Object o : (Object[]) value) list.add(formatArg(o));
            return list;
        }
        if (value instanceof Map) {
            Map<String, Object> m = new java.util.LinkedHashMap<>();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) value).entrySet()) {
                m.put(String.valueOf(e.getKey()), formatArg(e.getValue()));
            }
            return m;
        }
        if (value instanceof TensorVector) {
            List<Object> list = new ArrayList<>();
            for (Tensor t : extractTensors(value)) list.add(formatArg(t));
            return list;
        }
        Map<String, Object> fallback = new java.util.LinkedHashMap<>();
        fallback.put("_type", value.getClass().getSimpleName());
        String repr = String.valueOf(value);
        if (repr.length() > 50) repr = repr.substring(0, 50);
        fallback.put("repr", repr);
        return fallback;
    }

    public static List<Object> formatArgs(Object... args) {
        if (args == null) return Collections.emptyList();
        List<Object> out = new ArrayList<>(args.length);
        for (Object a : args) out.add(formatArg(a));
        return out;
    }
}
