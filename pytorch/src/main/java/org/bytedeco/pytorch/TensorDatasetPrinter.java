package org.bytedeco.pytorch;

import org.bytedeco.javacpp.LongPointer;

/**
 * Companion to {@link TensorPrinter} / {@link ModulePrinter} for
 * {@link TensorDataset}. The default {@code toString()} inherits the
 * verbose {@code org.bytedeco.javacpp.Pointer} address dump; the
 * override installed by {@code strip_module_forward.sh} delegates to
 * this class so {@code System.out.println(ds)} shows
 * {@code TensorDataset(num_samples=N, tensor.shape=[n,d], dtype=kFloat)}.
 */
final class TensorDatasetPrinter {

    private TensorDatasetPrinter() {}

    static String format(TensorDataset ds) {
        if (ds == null) return "null";
        StringBuilder sb = new StringBuilder("TensorDataset(");
        sb.append("num_samples=").append(longSize(ds));
        try {
            Tensor t = ds.tensor();
            sb.append(", tensor.shape=[");
            long[] shape = sizesOf(t);
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) sb.append(",");
                sb.append(shape[i]);
            }
            sb.append("],dtype=").append(t.scalar_type());
        } catch (Throwable e) {
            sb.append(", tensor=?");
        }
        sb.append(")");
        return sb.toString();
    }

    /** ds.size() returns SizeTOptional wrapping a long. */
    private static long longSize(TensorDataset ds) {
        try {
            org.bytedeco.pytorch.SizeTOptional opt = ds.size();
            if (opt == null) return -1;
            return opt.get();
        } catch (Throwable e) { return -1; }
    }

    private static long[] sizesOf(Tensor t) {
        try {
            org.bytedeco.pytorch.LongHeaderOnlyArrayRef ref = t.sizes();
            long n = ref.size();
            if (n == 0) return new long[0];
            return ref.vec().get();
        } catch (Throwable e) { return new long[0]; }
    }
}
