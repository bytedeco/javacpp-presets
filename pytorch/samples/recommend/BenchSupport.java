/*
 * Shared harness for recommend engineering benchmarks (pure Java, no native).
 */
package samples.recommend;

import java.util.Locale;
import java.util.Objects;
import java.util.Random;

final class BenchSupport {

    private BenchSupport() {}

    @FunctionalInterface
    interface CheckedRunnable {
        void run() throws Exception;
    }

    static final class Suite {
        final String name;
        int passed;
        int failed;
        final StringBuilder report = new StringBuilder();

        Suite(String name) {
            this.name = name;
        }

        void benchmark(String caseName, CheckedRunnable r) {
            long t0 = System.nanoTime();
            try {
                r.run();
                long ms = (System.nanoTime() - t0) / 1_000_000L;
                System.out.println("  OK  " + caseName + " (" + ms + " ms)");
            } catch (Throwable e) {
                failed++;
                long ms = (System.nanoTime() - t0) / 1_000_000L;
                System.out.println(" FAIL " + caseName + " (" + ms + " ms): " + e.getMessage());
                report.append("FAIL ").append(caseName).append(": ").append(e).append('\n');
                e.printStackTrace(System.out);
            }
        }

        void check(String label, boolean ok) {
            if (ok) {
                passed++;
            } else {
                failed++;
                report.append("CHECK FAILED: ").append(label).append('\n');
                System.out.println("    CHECK FAIL: " + label);
            }
        }

        void checkEq(String label, long expected, long actual) {
            boolean ok = expected == actual;
            if (!ok) {
                System.out.println("    " + label + ": expected " + expected + " got " + actual);
            }
            check(label, ok);
        }

        void checkEq(String label, Object expected, Object actual) {
            boolean ok = Objects.equals(expected, actual);
            if (!ok) {
                System.out.println("    " + label + ": expected " + expected + " got " + actual);
            }
            check(label, ok);
        }

        void checkClose(String label, double expected, double actual, double eps) {
            boolean ok = Math.abs(expected - actual) <= eps
                    || (Double.isNaN(expected) && Double.isNaN(actual));
            if (!ok) {
                System.out.printf(Locale.ROOT, "    %s: expected %.8f got %.8f (eps=%.8f)%n",
                        label, expected, actual, eps);
            }
            check(label, ok);
        }

        void checkRange(String label, double v, double lo, double hi) {
            boolean ok = v >= lo && v <= hi;
            if (!ok) {
                System.out.printf(Locale.ROOT, "    %s: %.8f not in [%.8f, %.8f]%n", label, v, lo, hi);
            }
            check(label, ok);
        }

        void checkTrue(String label, boolean ok) {
            check(label, ok);
        }

        int exitCode() {
            System.out.println("------------------------------------------");
            System.out.println(name + "  passed=" + passed + " failed=" + failed);
            if (failed > 0 && report.length() > 0) {
                System.out.println(report);
            }
            System.out.println(failed == 0 ? "ALL PASS" : "HAS FAILURES");
            return failed == 0 ? 0 : 1;
        }

        void header() {
            System.out.println("==========================================");
            System.out.println("  " + name);
            System.out.println("==========================================");
        }
    }

    static Random rng(long seed) {
        return new Random(seed);
    }

    static double[] randomGaussian(Random rng, int n, double mean, double std) {
        double[] a = new double[n];
        for (int i = 0; i < n; i++) {
            a[i] = mean + std * rng.nextGaussian();
        }
        return a;
    }

    static float[] randomFloat01(Random rng, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = rng.nextFloat();
        }
        return a;
    }

    static float[] randomBinary(Random rng, int n, double p) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = rng.nextDouble() < p ? 1.0f : 0.0f;
        }
        return a;
    }

    static String[] userIds(int n, int cardinality, Random rng) {
        String[] ids = new String[n];
        for (int i = 0; i < n; i++) {
            ids[i] = "u" + rng.nextInt(Math.max(1, cardinality));
        }
        return ids;
    }
}
