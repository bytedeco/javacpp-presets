package org.bytedeco.pytorch;

/**
 * Companion printer for {@link LossReduction} (c10 enum). Default
 * toString is the verbose Pointer address dump. We render just the
 * JavaCPP-generated enum constant name (e.g. {@code kMean},
 * {@code kSum}, {@code kNone}).
 *
 * Implementation: LossReduction is a std::variant&lt;kNone,kMean,kSum&gt;.
 * Use the {@code get0/1/2} accessors to test which variant is active.
 * The active accessor returns a valid enum value; the others return
 * a "null-equivalent" (e.g. an uninitialized kMean) — but to be safe
 * we just try all three and emit the first non-null / non-error result.
 */
final class LossReductionPrinter {

    private LossReductionPrinter() {}

    static String format(LossReduction r) {
        if (r == null) return "null";
        try {
            // Try each variant — the active one will return its enum
            // value without crashing, others may throw.
            if (r.get0() != null) return "kNone";
        } catch (Throwable ignored) {}
        try {
            if (r.get1() != null) return "kMean";
        } catch (Throwable ignored) {}
        try {
            if (r.get2() != null) return "kSum";
        } catch (Throwable ignored) {}
        return "?";
    }
}
