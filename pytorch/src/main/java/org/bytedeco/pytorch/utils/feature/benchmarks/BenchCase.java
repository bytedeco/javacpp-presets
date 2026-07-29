/*
 * Shared bench result + helpers for feature platform benchmarks.
 */
package org.bytedeco.pytorch.utils.feature.benchmarks;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** One benchmark case outcome. */
public final class BenchCase {

    public final String name;
    public final boolean passed;
    public final String detail;
    public final double elapsedMs;
    public final Map<String, Object> metrics;

    public BenchCase(String name, boolean passed, String detail, double elapsedMs,
                     Map<String, Object> metrics) {
        this.name = name;
        this.passed = passed;
        this.detail = detail != null ? detail : "";
        this.elapsedMs = elapsedMs;
        this.metrics = metrics != null ? new LinkedHashMap<>(metrics) : Map.of();
    }

    public static BenchCase pass(String name, String detail, long nanos) {
        return new BenchCase(name, true, detail, nanos / 1_000_000.0, null);
    }

    public static BenchCase pass(String name, String detail, long nanos, Map<String, Object> metrics) {
        return new BenchCase(name, true, detail, nanos / 1_000_000.0, metrics);
    }

    public static BenchCase fail(String name, String detail, long nanos) {
        return new BenchCase(name, false, detail, nanos / 1_000_000.0, null);
    }

    @Override
    public String toString() {
        String status = passed ? "PASS" : "FAIL";
        return String.format(Locale.ROOT, "[%s] %-28s %8.2f ms  %s", status, name, elapsedMs, detail);
    }

    /** Accumulator for a suite run. */
    public static final class Suite {
        public final String name;
        public final List<BenchCase> cases = new ArrayList<>();

        public Suite(String name) {
            this.name = name;
        }

        public void add(BenchCase c) {
            cases.add(c);
            System.out.println(c);
        }

        public int passed() {
            int n = 0;
            for (BenchCase c : cases) if (c.passed) n++;
            return n;
        }

        public int failed() {
            return cases.size() - passed();
        }

        public boolean ok() {
            return failed() == 0;
        }

        public void summary() {
            System.out.printf(Locale.ROOT, "%n=== %s: %d/%d passed, %d failed ===%n",
                    name, passed(), cases.size(), failed());
        }
    }
}
