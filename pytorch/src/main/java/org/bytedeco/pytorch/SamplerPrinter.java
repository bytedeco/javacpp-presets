package org.bytedeco.pytorch;

import org.bytedeco.javacpp.LongPointer;

/**
 * Companion printer for {@link Sampler} subclasses (RandomSampler,
 * SequentialSampler, BatchSizeSampler, WeightedRandomSampler, ...).
 * Samplers carry a length (the dataset size) and optionally an index
 * dtype; BatchSizeSampler wraps any inner sampler with a target
 * batch size. We print the relevant fields via reflection.
 */
final class SamplerPrinter {

    private SamplerPrinter() {}

    static String format(Sampler s) {
        if (s == null) return "null";
        StringBuilder sb = new StringBuilder();
        String cls = getClassSimpleName(s);
        sb.append(cls).append("(");
        appendLongIfPresent(sb, s, "size", "size=");
        // RandomSampler has a public `size` only; BatchSizeSampler
        // adds a `batch_size`.
        appendLongIfPresent(sb, s, "batch_size", ", batch_size=");
        sb.append(")");
        return sb.toString();
    }

    private static String getClassSimpleName(Object o) {
        try {
            return o.getClass().getSimpleName();
        } catch (Throwable e) {
            return "Sampler";
        }
    }

    private static void appendLongIfPresent(StringBuilder sb,
                                           Object o,
                                           String method,
                                           String prefix) {
        try {
            java.lang.reflect.Method m = o.getClass().getMethod(method);
            Object v = m.invoke(o);
            long n;
            if (v instanceof LongPointer) {
                n = ((LongPointer) v).get(0);
            } else if (v instanceof Number) {
                n = ((Number) v).longValue();
            } else {
                return;
            }
            sb.append(prefix).append(n);
        } catch (Throwable ignored) {
            // method doesn't exist on this sampler; skip.
        }
    }
}
