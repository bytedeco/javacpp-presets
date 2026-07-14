package org.bytedeco.pytorch;

import org.bytedeco.javacpp.LongPointer;

/**
 * Companion printer for {@link FullDataLoaderOptions} (extends
 * DataLoaderOptions with workers, max_jobs, timeout,
 * enforce_ordering, etc.).
 */
final class FullDataLoaderOptionsPrinter {

    private FullDataLoaderOptionsPrinter() {}

    static String format(FullDataLoaderOptions o) {
        if (o == null) return "null";
        StringBuilder sb = new StringBuilder("FullDataLoaderOptions(");
        appendLong(sb, o, "batch_size", "batch_size=");
        appendLong(sb, o, "workers", ", workers=");
        appendLong(sb, o, "max_jobs", ", max_jobs=");
        // timeout() returns std::optional<Milliseconds>; treat 0 as not set.
        try {
            java.lang.reflect.Method m = o.getClass().getMethod("timeout");
            Object v = m.invoke(o);
            // Skip printing timeout if it's a not-set optional (the
            // Milliseconds wrapper around 0 is indistinguishable from a
            // default 0; we just print what we have).
            sb.append(", timeout=").append(v);
        } catch (Throwable ignored) {}
        appendBool(sb, o, "enforce_ordering", ", enforce_ordering=");
        appendBool(sb, o, "drop_last", ", drop_last=");
        sb.append(")");
        return sb.toString();
    }

    private static void appendLong(StringBuilder sb, Object o,
                                   String method, String prefix) {
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
        } catch (Throwable ignored) { }
    }

    private static void appendBool(StringBuilder sb, Object o,
                                    String method, String prefix) {
        try {
            java.lang.reflect.Method m = o.getClass().getMethod(method);
            Object v = m.invoke(o);
            boolean b;
            if (v instanceof org.bytedeco.javacpp.BoolPointer) {
                b = ((org.bytedeco.javacpp.BoolPointer) v).get(0);
            } else if (v instanceof Number) {
                b = ((Number) v).intValue() != 0;
            } else if (v instanceof Boolean) {
                b = (Boolean) v;
            } else {
                return;
            }
            sb.append(prefix).append(b);
        } catch (Throwable ignored) { }
    }
}
