package org.bytedeco.pytorch;

import org.bytedeco.javacpp.BoolPointer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.ShortPointer;

/**
 * Mirrors Python PyTorch's print(tensor) for JavaCPP Tensor. Without
 * this, System.out.println(tensor) prints a memory address. Output:
 *
 *   Tensor[shape=[2, 3], dtype=kFloat, device=cpu, numel=6, values=
 *     [[ 0.1230, -0.4560,  0.7890],
 *      [ 1.2340,  5.6780, -9.0120]]]
 *
 * Uses TensorBase.data_ptr_<dtype>() to bulk-read the elements into
 * a Java array in a single JNI call, then formats nested by shape.
 * The per-element readScalar uses .get(flatIndex) only on a slice of
 * the bulk array (which has the correct length) — calling .get() on
 * the raw FloatPointer alone is limited by FloatPointer's own
 * position/length metadata (defaults to 0/2).
 */
final class TensorPrinter {

    /** Above this many elements, the values are truncated to head/tail. */
    private static final int PRINT_LIMIT = 256;
    /** Number of head/tail elements when truncated. */
    private static final int HEAD_TAIL = 4;

    private TensorPrinter() {}

    static String format(Tensor t) {
        if (t == null || t.isNull()) return "Tensor[null]";

        StringBuilder sb = new StringBuilder("Tensor[");
        sb.append("shape=").append(formatShape(t));
        sb.append(", dtype=").append(formatDtype(t));
        sb.append(", device=").append(formatDevice(t));

        long n;
        try {
            n = t.numel();
        } catch (Throwable e) {
            sb.append("]");
            return sb.toString();
        }
        sb.append(", numel=").append(n);

        if (n == 0) {
            sb.append("]");
            return sb.toString();
        }

        sb.append(", values=\n");
        try {
            sb.append(indent(formatValues(t, (int) n))).append(']');
        } catch (Throwable e) {
            sb.append("  <error printing values: ").append(e).append(">]");
        }
        return sb.toString();
    }

    private static String formatShape(Tensor t) {
        try {
            long[] sizes = sizesAsArray(t.sizes());
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < sizes.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(sizes[i]);
            }
            sb.append(']');
            return sb.toString();
        } catch (Throwable e) {
            return "?";
        }
    }

    private static String formatDtype(Tensor t) {
        try {
            String s = String.valueOf(t.scalar_type());
            int dot = s.lastIndexOf('.');
            return (dot >= 0 ? s.substring(dot + 1) : s);
        } catch (Throwable e) {
            return "?";
        }
    }

    private static String formatDevice(Tensor t) {
        try {
            // Use the typed Device accessors (is_cuda, is_cpu, type)
            // rather than parsing toString, which is verbose and doesn't
            // always include "type=" for the libtorch 2.12 build we have.
            Object dev = t.device();
            if (dev == null) return "?";
            if (isCudaDevice(dev)) {
                return cudaDeviceString(dev);
            }
            if (isCpuDevice(dev)) return "cpu";
            // Fall back to toString of the DeviceType field.
            return String.valueOf(dev);
        } catch (Throwable e) {
            return "?";
        }
    }

    private static boolean isCudaDevice(Object dev) {
        try {
            return (boolean) dev.getClass().getMethod("is_cuda").invoke(dev);
        } catch (Throwable e) { return false; }
    }

    private static boolean isCpuDevice(Object dev) {
        try {
            return (boolean) dev.getClass().getMethod("is_cpu").invoke(dev);
        } catch (Throwable e) { return false; }
    }

    private static String cudaDeviceString(Object dev) {
        try {
            Object type = dev.getClass().getMethod("type").invoke(dev);
            // type is a DeviceType enum; its toString is "CUDA" (or
            // "HIP", "MPS" for other backends).
            String s = String.valueOf(type);
            // For cuda:0 vs cuda:1 the index matters; for CPU it's
            // just "cpu".
            int idx = (int) (byte) dev.getClass().getMethod("index").invoke(dev);
            String sLower = s.toLowerCase();
            return sLower + ":" + idx;
        } catch (Throwable e) { return "cuda"; }
    }

    private static long[] sizesAsArray(org.bytedeco.pytorch.LongHeaderOnlyArrayRef ref) {
        long len = ref.size();
        if (len == 0) return new long[0];
        return ref.vec().get();
    }

    /**
     * Formats the values of the contiguous tensor nested by shape. The
     * per-element read uses Tensor.cpu().item_*() for safety (the typed
     * data_ptr_X accessors return JavaCPP wrapper pointers whose
     * position/limit metadata is sometimes wrong — the ByteBuffer view
     * trick didn't help either). Per-element access is slower but
     * always correct, and PRINT_LIMIT keeps it usable.
     */
    private static String formatValues(Tensor t, int numelTotal) {
        Tensor contig = null;
        try {
            contig = t.contiguous();
        } catch (Throwable e) {
            return "  <contiguous() failed: " + e + ">";
        }
        String dtype = formatDtype(contig);
        long[] shape = sizesAsArray(contig.sizes());
        try {
            int n = numelTotal;
            if (n <= PRINT_LIMIT) {
                return nest(contig, dtype, shape, 0, 0, n);
            }
            int headCount = Math.max(1, Math.min(HEAD_TAIL, n / 2));
            int tailStart = n - HEAD_TAIL;
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < headCount; i++) {
                if (i > 0) sb.append(", ");
                sb.append(formatScalar(contig, dtype, i));
            }
            sb.append(", ..., ...,\n  ");
            for (int i = tailStart; i < n; i++) {
                if (i > tailStart) sb.append(", ");
                sb.append(formatScalar(contig, dtype, i));
            }
            return sb.toString();
        } finally {
            // JavaCPP refcount handles it.
        }
    }

    private static String nest(Tensor t, String dtype, long[] shape,
                               int dim, int offset, int count) {
        if (shape.length == 0 || dim == shape.length - 1) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < count; i++) {
                if (i > 0) sb.append(", ");
                sb.append(formatScalar(t, dtype, offset + i));
            }
            sb.append(']');
            return sb.toString();
        }
        int innerCount = (int) shape[dim + 1];
        long stride = 1;
        for (int i = dim + 1; i < shape.length; i++) stride *= shape[i];
        int outerRows = Math.min(
            (int) ((count + innerCount - 1) / innerCount),
            (int) shape[dim]);
        StringBuilder sb = new StringBuilder("[");
        for (int r = 0; r < outerRows; r++) {
            if (r > 0) sb.append(",\n ");
            int subOffset = offset + (int) (r * stride);
            int subCount = Math.min(innerCount, count - r * innerCount);
            sb.append(nest(t, dtype, shape, dim + 1, subOffset, subCount));
        }
        sb.append(']');
        return sb.toString();
    }

    /**
     * Format a single scalar element from the tensor at flatIndex.
     * Uses view+select to navigate to the scalar position, then
     * item_* to read it. Limit PRINT_LIMIT keeps the cost reasonable.
     */
    private static String formatScalar(Tensor t, String dtype, int flatIndex) {
        try {
            Tensor scalar;
            try {
                // flat.view(-1) flattens to 1D; .select(0, flatIndex) picks the
                // flatIndex-th element.
                scalar = t.contiguous().view(new long[]{-1L}).select(0, flatIndex);
            } catch (Throwable e) {
                return "<err>";
            }
            try {
                if (dtype.equals("kFloat") || dtype.equals("Float")
                        || dtype.equals("kDouble") || dtype.equals("Double")
                        || dtype.equals("kHalf") || dtype.equals("Half")
                        || dtype.equals("kBFloat16") || dtype.equals("BFloat16")) {
                    return String.format("%.4f", scalar.item_double());
                }
                if (dtype.equals("kLong") || dtype.equals("Long")) {
                    return Long.toString(scalar.item_long());
                }
                if (dtype.equals("kInt") || dtype.equals("Int")) {
                    return Integer.toString(scalar.item_int());
                }
                if (dtype.equals("kBool") || dtype.equals("Bool")) {
                    return scalar.item_bool() ? "true" : "false";
                }
                if (dtype.equals("kShort") || dtype.equals("Short")) {
                    return Integer.toString(scalar.item_short());
                }
                if (dtype.equals("kByte") || dtype.equals("Byte")
                        || dtype.equals("kChar") || dtype.equals("Char")) {
                    return Byte.toString(scalar.item_char());
                }
                return "<unhandled-dtype:" + dtype + ">";
            } finally {
                // JavaCPP refcount handles it.
            }
        } catch (Throwable e) {
            return "<err>";
        }
    }

    private static String indent(String s) {
        StringBuilder sb = new StringBuilder();
        for (String line : s.split("\n")) {
            sb.append("  ").append(line).append('\n');
        }
        if (sb.length() > 0 && sb.charAt(sb.length() - 1) == '\n') {
            sb.setLength(sb.length() - 1);
        }
        return sb.toString();
    }
}
