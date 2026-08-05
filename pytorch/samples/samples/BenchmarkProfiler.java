package samples;
import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.profiler.*;

import org.bytedeco.pytorch.profiler.ActivityTypeSet;
import org.bytedeco.pytorch.profiler.ExperimentalConfig;
import org.bytedeco.pytorch.profiler.ProfilerConfig;
import org.bytedeco.pytorch.profiler.ProfilerResult;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.bytedeco.pytorch.global.torch.ActivityType;
import static org.bytedeco.pytorch.global.torch.ProfilerState;
import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.disableProfiler;
import static org.bytedeco.pytorch.global.torch.enableProfiler;
import static org.bytedeco.pytorch.global.torch.mm;
import static org.bytedeco.pytorch.global.torch.prepareProfiler;
import static org.bytedeco.pytorch.global.torch.randn;

/**
 * Benchmark / smoke test for the Kineto chrome-trace path after JavaCPP re-parse:
 * {@code enableProfiler} → work → {@code disableProfiler} → {@link ProfilerResult#save}.
 *
 * <p>Does <b>not</b> use {@code exportMemoryProfile} (Python-only NoOp under pure JavaCPP).
 */
public class BenchmarkProfiler {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name);
        } else {
            failed++;
            System.out.println("  FAIL  " + name);
        }
    }

    static void section(String t) {
        System.out.println("\n=== " + t + " ===");
    }

    public static void main(String[] args) throws Exception {
        section("Kineto enableProfiler / disableProfiler / save");

        Path outDir = Path.of("samples/out/profiler_bench");
        Files.createDirectories(outDir);
        String exportPath = outDir.resolve("kineto_trace.json").toAbsolutePath().toString();
        // remove stale
        new File(exportPath).delete();

        ExperimentalConfig experimental = new ExperimentalConfig();
        experimental.verbose(true);
        experimental.adjust_timestamps(true);
        experimental.disable_external_correlation(true);

        BytePointer traceId = new BytePointer("benchmark_profiler");
        ProfilerConfig config = new ProfilerConfig(
                ProfilerState.KINETO,
                true,  // report_input_shapes
                true,  // profile_memory
                false, // with_stack
                true,  // with_flops
                false, // with_modules
                experimental,
                traceId
        );

        ActivityTypeSet activities = new ActivityTypeSet();
        activities.insert(ActivityType.CPU);

        long t0 = System.nanoTime();
        prepareProfiler(config, activities);
        enableProfiler(config, activities);

        TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        Tensor a = randn(new long[]{512, 512}, opts);
        Tensor b = randn(new long[]{512, 512}, opts);
        Tensor c = mm(a, b);
        c.add_(new Scalar(1.0));
        c.cpu().data_ptr();

        ProfilerResult result = disableProfiler();
        long t1 = System.nanoTime();
        check("disableProfiler non-null", result != null && !result.isNull());

        if (result != null && !result.isNull()) {
            result.save(exportPath);
            result.close();
        }

        File f = new File(exportPath);
        check("trace file exists", f.exists());
        check("trace file non-empty", f.exists() && f.length() > 0);
        if (f.exists()) {
            System.out.println("  file: " + f.getAbsolutePath() + " (" + f.length() + " bytes)");
            // chrome-trace JSON usually starts with { or [
            String head = Files.readString(f.toPath()).substring(0, Math.min(80, (int) f.length())).trim();
            check("looks like JSON", head.startsWith("{") || head.startsWith("["));
        }
        System.out.printf("  wall_ms=%.1f%n", (t1 - t0) / 1e6);

        section("ProcessGroupNativeWrapper class present");
        try {
            Class<?> cl = Class.forName("org.bytedeco.pytorch.distributed.ProcessGroupNativeWrapper");
            check("ProcessGroupNativeWrapper loaded", cl != null);
            System.out.println("  class: " + cl.getName());
        } catch (ClassNotFoundException e) {
            check("ProcessGroupNativeWrapper loaded", false);
            System.out.println("  (re-run JavaCPP parse if missing)");
        }

        section("ProfilerResult / enableProfiler symbols");
        try {
            Class<?> pr = Class.forName("org.bytedeco.pytorch.profiler.ProfilerResult");
            check("ProfilerResult class", pr != null);
            var m = org.bytedeco.pytorch.global.torch.class.getMethod(
                    "enableProfiler",
                    ProfilerConfig.class,
                    ActivityTypeSet.class);
            check("enableProfiler method", m != null);
            var m2 = org.bytedeco.pytorch.global.torch.class.getMethod("disableProfiler");
            check("disableProfiler method", m2 != null);
        } catch (Throwable e) {
            check("profiler symbols", false);
            e.printStackTrace(System.out);
        }

        System.out.println("\n==============================");
        System.out.println("Passed: " + passed + "  Failed: " + failed);
        System.out.println("==============================");
        if (failed > 0) {
            System.exit(1);
        }
    }
}
