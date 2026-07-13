package org.bytedeco.pytorch;

import org.bytedeco.javacpp.DoublePointer;

/**
 * Companion to {@link ModulePrinter} / {@link TensorPrinter} for
 * Optimizer subclasses (Adam, SGD, AdamW, LBFGS, RMSprop, Adagrad).
 * The default {@code toString()} inherits the verbose
 * {@code org.bytedeco.javacpp.Pointer} address dump; the override
 * installed by {@code strip_module_forward.sh} delegates here so
 * {@code System.out.println(opt)} prints the param groups with their
 * per-group hyper-params (lr, betas, weight_decay, etc.).
 */
final class OptimizerPrinter {

    private OptimizerPrinter() {}

    static String format(Optimizer opt) {
        if (opt == null) return "null";
        StringBuilder sb = new StringBuilder();
        String cls = getClassSimpleName(opt);
        sb.append(cls);
        try {
            long n = opt.size();
            sb.append("(num_param_groups=").append(n);
        } catch (Throwable e) {
            sb.append("(num_param_groups=?");
        }
        try {
            long nParams = opt.parameters().size();
            sb.append(", num_params=").append(nParams);
        } catch (Throwable e) {
            sb.append(", num_params=?");
        }
        try {
            sb.append(", defaults=");
            OptimizerOptions defaults = opt.defaults();
            sb.append(formatOptions(defaults));
        } catch (Throwable e) {
            sb.append(", defaults=?");
        }
        sb.append(", groups=[");
        try {
            OptimizerParamGroupVector pgs = opt.param_groups();
            int n = (int) Math.min(pgs.size(), 8L);
            for (int i = 0; i < n; i++) {
                if (i > 0) sb.append(", ");
                OptimizerParamGroup pg = pgs.get(i);
                sb.append("(");
                try {
                    long pn = pg.params().size();
                    sb.append("n_params=").append(pn);
                } catch (Throwable e) {
                    sb.append("n_params=?");
                }
                sb.append(", ");
                sb.append(formatOptions(pg.options()));
                sb.append(")");
            }
            if (pgs.size() > n) sb.append(", …");
        } catch (Throwable e) {
            sb.append("?");
        }
        sb.append("])");
        return sb.toString();
    }

    private static String getClassSimpleName(Object o) {
        try {
            String cn = o.getClass().getSimpleName();
            return cn;
        } catch (Throwable e) {
            return "Optimizer";
        }
    }

    /**
     * Print the options struct. Optimizer options all derive from
     * OptimizerOptions which exposes {@code get_lr()} and an "is"
     * accessor pattern (JavaCPP generated). We use reflection so the
     * same code works for Adam, AdamW, SGD, Adagrad, RMSprop, LBFGS.
     */
    private static String formatOptions(OptimizerOptions opts) {
        if (opts == null) return "Options(null)";
        StringBuilder sb = new StringBuilder();
        sb.append("lr=");
        try { sb.append(String.format("%.4g", opts.get_lr())); }
        catch (Throwable e) {
            try {
                java.lang.reflect.Method m = opts.getClass().getMethod("get_lr");
                sb.append(String.format("%.4g", ((Number) m.invoke(opts)).doubleValue()));
            } catch (Throwable e2) { sb.append("?"); }
        }
        appendDoubleOpt(sb, opts, "betas");
        appendDoubleOpt(sb, opts, "momentum");
        appendDoubleOpt(sb, opts, "weight_decay");
        appendDoubleOpt(sb, opts, "dampening");
        appendBoolOpt(sb, opts, "nesterov");
        appendBoolOpt(sb, opts, "amsgrad");
        return sb.toString();
    }

    private static void appendDoubleOpt(StringBuilder sb,
                                       OptimizerOptions opts,
                                       String fieldName) {
        try {
            // For betas which is std::tuple<double, double>, JavaCPP
            // exposes a method named get_betas() returning a
            // DoublePointer (length 2). For weight_decay etc. it's
            // just a single double getter.
            java.lang.reflect.Method m = opts.getClass().getMethod("get_" + fieldName);
            Object v = m.invoke(opts);
            if (v instanceof DoublePointer) {
                DoublePointer p = (DoublePointer) v;
                sb.append(", ").append(fieldName).append("=(");
                int n = (int) p.capacity();
                for (int i = 0; i < n; i++) {
                    if (i > 0) sb.append(",");
                    sb.append(String.format("%.4g", p.get(i)));
                }
                sb.append(")");
            } else if (v instanceof Number) {
                sb.append(", ").append(fieldName).append("=");
                sb.append(String.format("%.4g", ((Number) v).doubleValue()));
            } else if (v != null) {
                sb.append(", ").append(fieldName).append("=").append(v);
            }
        } catch (Throwable ignored) {
            // Field doesn't exist on this optimizer's options; skip.
        }
    }

    private static void appendBoolOpt(StringBuilder sb,
                                      OptimizerOptions opts,
                                      String fieldName) {
        try {
            java.lang.reflect.Method m = opts.getClass().getMethod(fieldName);
            sb.append(", ").append(fieldName).append("=").append(m.invoke(opts));
        } catch (Throwable ignored) { }
    }
}
