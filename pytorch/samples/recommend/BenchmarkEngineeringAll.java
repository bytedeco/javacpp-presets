/*
 * Master runner for all recommend engineering benchmarks (pure Java).
 *
 * Runs modules in sequence and prints a summary. Each module is also
 * independently runnable via its own main().
 *
 *   java -cp target/classes:samples:... samples.recommend.BenchmarkEngineeringAll
 *
 * Or after compiling samples + engineering sources to a single classes dir.
 */
package samples.recommend;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

public final class BenchmarkEngineeringAll {

    private static final class ModuleSpec {
        final String name;
        final Class<?> mainClass;

        ModuleSpec(String name, Class<?> mainClass) {
            this.name = name;
            this.mainClass = mainClass;
        }
    }

    public static void main(String[] args) {
        System.out.println("##################################################");
        System.out.println("#  Recommend Engineering Benchmarks (ALL)        #");
        System.out.println("##################################################");

        List<ModuleSpec> modules = List.of(
                new ModuleSpec("abtest", BenchmarkAbtest.class),
                new ModuleSpec("offline", BenchmarkOffline.class),
                new ModuleSpec("pipeline", BenchmarkPipeline.class),
                new ModuleSpec("deploy", BenchmarkDeploy.class),
                new ModuleSpec("gateway", BenchmarkGateway.class),
                new ModuleSpec("ops", BenchmarkOps.class),
                new ModuleSpec("modelops", BenchmarkModelOps.class));

        // Optional filter: --only=abtest,ops
        String only = null;
        for (String a : args) {
            if (a.startsWith("--only=")) {
                only = a.substring("--only=".length());
            }
        }

        int passModules = 0;
        int failModules = 0;
        List<String> failedNames = new ArrayList<>();
        long tAll = System.nanoTime();

        for (ModuleSpec m : modules) {
            if (only != null && !containsToken(only, m.name)) {
                System.out.println("\n[SKIP] " + m.name);
                continue;
            }
            System.out.println("\n######## MODULE: " + m.name + " (" + m.mainClass.getSimpleName() + ") ########");
            long t0 = System.nanoTime();
            int code = runModuleMain(m.mainClass);
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            if (code == 0) {
                passModules++;
                System.out.println("[MODULE PASS] " + m.name + " (" + ms + " ms)");
            } else {
                failModules++;
                failedNames.add(m.name);
                System.out.println("[MODULE FAIL] " + m.name + " exit=" + code + " (" + ms + " ms)");
            }
        }

        long totalMs = (System.nanoTime() - tAll) / 1_000_000L;
        System.out.println("\n##################################################");
        System.out.println("#  SUMMARY  passModules=" + passModules
                + " failModules=" + failModules
                + " totalMs=" + totalMs);
        if (!failedNames.isEmpty()) {
            System.out.println("#  failed: " + failedNames);
        }
        System.out.println("##################################################");
        System.exit(failModules == 0 ? 0 : 1);
    }

    /**
     * Invoke module main without letting System.exit kill the JVM.
     * Installs a SecurityManager-free approach: call a package-visible run if
     * present; otherwise redirect and invoke main via a wrapper that catches
     * the exit by running in-process checks through reflection on a synthetic
     * runner.
     *
     * <p>Simpler approach used here: each benchmark's logic is executed by
     * re-calling {@code main} but we replace {@link System#exit} effect by
     * running under a custom {@code ExitTrap} via security is deprecated, so
     * we instead duplicate invocation by expecting benchmarks to throw
     * {@link ExitException} — not available.
     *
     * <p>Practical approach: run each module's main in a fresh way — catch
     * nothing from exit. We refactor to call {@code run(Suite)} pattern...
     * For zero intrusion, use a subprocess... but keep in-process:
     * intercept by checking that modules use {@code System.exit} — we will
     * invoke the public static methods if we change them.
     *
     * <p>Implemented: use {@link #runInProcess} which launches main and
     * relies on benchmarks being converted to return codes via
     * {@link RunnableMain}. Fallback: reflective main under
     * {@link NoExitSecurity} disabled on modern JDK — so we run modules
     * as separate logic by copying entry to {@code run(String[])} returning int.
     */
    private static int runModuleMain(Class<?> cls) {
        try {
            Method runTests = cls.getMethod("runTests");
            if (runTests.getReturnType() == int.class) {
                Object r = runTests.invoke(null);
                return ((Integer) r).intValue();
            }
        } catch (NoSuchMethodException e) {
            return runAsSubprocess(cls.getName());
        } catch (Exception e) {
            e.printStackTrace(System.out);
            return 2;
        }
        return runAsSubprocess(cls.getName());
    }

    private static int runAsSubprocess(String className) {
        try {
            String cp = System.getProperty("java.class.path");
            ProcessBuilder pb = new ProcessBuilder(
                    System.getProperty("java.home") + "/bin/java",
                    "-cp", cp,
                    className);
            pb.redirectErrorStream(true);
            Process p = pb.start();
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            p.getInputStream().transferTo(bos);
            int code = p.waitFor();
            System.out.print(bos.toString());
            return code;
        } catch (Exception e) {
            e.printStackTrace(System.out);
            return 2;
        }
    }

    private static boolean containsToken(String csv, String token) {
        for (String t : csv.split(",")) {
            if (t.trim().equalsIgnoreCase(token)) return true;
        }
        return false;
    }
}
