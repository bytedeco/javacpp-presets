package org.bytedeco.pytorch;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.BoolPointer;

/**
 * Companion printer for DataLoaderOptions / FullDataLoaderOptions.
 * Shows batch_size, num_workers, drop_last, batch_sampler, plus
 * the backend enum and pin_memory flag.
 */
final class DataLoaderConfigPrinter {

    private DataLoaderConfigPrinter() {}

    static String format(DataLoaderOptions opts) {
        if (opts == null) return "null";
        StringBuilder sb = new StringBuilder("DataLoaderOptions(");
        appendDataLoaderOptions(sb, opts);
        sb.append(")");
        return sb.toString();
    }

    static String format(FullDataLoaderOptions opts) {
        if (opts == null) return "null";
        StringBuilder sb = new StringBuilder("FullDataLoaderOptions(");
        appendDataLoaderOptions(sb, opts);
        sb.append(")");
        return sb.toString();
    }

    private static void appendDataLoaderOptions(StringBuilder sb,
                                                org.bytedeco.javacpp.Pointer opts) {
        Class<?> c = opts.getClass();
        invokeLong(sb, c, opts, "batch_size", "batch_size=");
        invokeLong(sb, c, opts, "num_workers", ", num_workers=");
        invokeBool(sb, c, opts, "drop_last", ", drop_last=");
        invokeStr(sb, c, opts, "batch_sampler_type", ", batch_sampler=");
        invokeBool(sb, c, opts, "pin_memory", ", pin_memory=");
    }

    private static void invokeLong(StringBuilder sb, Class<?> c,
                                    Object opts, String method, String prefix) {
        try {
            java.lang.reflect.Method m = c.getMethod(method);
            // JavaCPP exposes size_t values via SizeTPointer (or
            // LongPointer, depending on the binding). Read the first
            // element via whichever pointer type the return is.
            Object v = m.invoke(opts);
            if (v instanceof org.bytedeco.javacpp.SizeTPointer) {
                sb.append(prefix).append(((org.bytedeco.javacpp.SizeTPointer) v).get(0));
            } else if (v instanceof org.bytedeco.javacpp.LongPointer) {
                sb.append(prefix).append(((org.bytedeco.javacpp.LongPointer) v).get(0));
            } else if (v instanceof Number) {
                sb.append(prefix).append(((Number) v).longValue());
            } else {
                sb.append(prefix).append(v);
            }
        } catch (Throwable ignored) { }
    }

    private static void invokeBool(StringBuilder sb, Class<?> c,
                                    Object opts, String method, String prefix) {
        try {
            java.lang.reflect.Method m = c.getMethod(method);
            Object v = m.invoke(opts);
            boolean b;
            if (v instanceof org.bytedeco.javacpp.BoolPointer) {
                // BoolPointer.get returns boolean. Compare to 1L to avoid
                // boolean != int mismatch.
                b = ((org.bytedeco.javacpp.BoolPointer) v).get(0);
            } else if (v instanceof Number) {
                b = ((Number) v).intValue() != 0;
            } else if (v instanceof Boolean) {
                b = (Boolean) v;
            } else {
                b = false;
            }
            sb.append(prefix).append(b);
        } catch (Throwable ignored) { }
    }

    private static void invokeStr(StringBuilder sb, Class<?> c,
                                   Object opts, String method, String prefix) {
        try {
            java.lang.reflect.Method m = c.getMethod(method);
            sb.append(prefix).append(m.invoke(opts));
        } catch (Throwable ignored) { }
    }
}
