package dataframe;
//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//


import java.io.File;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ActivityType;
import org.bytedeco.pytorch.global.torch.ProfilerState;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.profiler.ActivityTypeSet;
import org.bytedeco.pytorch.profiler.ExperimentalConfig;
import org.bytedeco.pytorch.profiler.ProfilerConfig;
import org.bytedeco.pytorch.profiler.ProfilerResult;

public class BenchmarkProfiler {
    static int passed = 0;
    static int failed = 0;

    public BenchmarkProfiler() {
    }

    static void check(String var0, boolean var1) {
        if (var1) {
            ++passed;
            System.out.println("  PASS  " + var0);
        } else {
            ++failed;
            System.out.println("  FAIL  " + var0);
        }

    }

    static void section(String var0) {
        System.out.println("\n=== " + var0 + " ===");
    }

    public static void main(String[] var0) throws Exception {
        section("Kineto enableProfiler / disableProfiler / save");
        Path var1 = Path.of("samples/out/profiler_bench");
        Files.createDirectories(var1);
        String var2 = var1.resolve("kineto_trace.json").toAbsolutePath().toString();
        (new File(var2)).delete();
        ExperimentalConfig var3 = new ExperimentalConfig();
        var3.verbose(true);
        var3.adjust_timestamps(true);
        var3.disable_external_correlation(true);
        BytePointer var4 = new BytePointer("benchmark_profiler");
        ProfilerConfig var5 = new ProfilerConfig(ProfilerState.KINETO, true, true, false, true, false, var3, var4);
        ActivityTypeSet var6 = new ActivityTypeSet();
        var6.insert(ActivityType.CPU);
        long var7 = System.nanoTime();
        torch.prepareProfiler(var5, var6);
        torch.enableProfiler(var5, var6);
        TensorOptions var9 = (new TensorOptions()).dtype(new ScalarTypeOptional(ScalarType.Float));
        Tensor var10 = torch.randn(new long[]{512L, 512L}, var9);
        Tensor var11 = torch.randn(new long[]{512L, 512L}, var9);
        Tensor var12 = torch.mm(var10, var11);
        var12.add_(new Scalar((double)1.0F));
        var12.cpu().data_ptr();
        ProfilerResult var13 = torch.disableProfiler();
        long var14 = System.nanoTime();
        check("disableProfiler non-null", var13 != null && !var13.isNull());
        if (var13 != null && !var13.isNull()) {
            var13.save(var2);
            var13.close();
        }

        File var16 = new File(var2);
        check("trace file exists", var16.exists());
        check("trace file non-empty", var16.exists() && var16.length() > 0L);
        if (var16.exists()) {
            PrintStream var10000 = System.out;
            String var10001 = var16.getAbsolutePath();
            var10000.println("  file: " + var10001 + " (" + var16.length() + " bytes)");
            String var17 = Files.readString(var16.toPath()).substring(0, Math.min(80, (int)var16.length())).trim();
            check("looks like JSON", var17.startsWith("{") || var17.startsWith("["));
        }

        System.out.printf("  wall_ms=%.1f%n", (double)(var14 - var7) / (double)1000000.0F);
        section("ProcessGroupNativeWrapper class present");

        try {
            Class var22 = Class.forName("org.bytedeco.pytorch.distributed.ProcessGroupNativeWrapper");
            check("ProcessGroupNativeWrapper loaded", var22 != null);
            System.out.println("  class: " + var22.getName());
        } catch (ClassNotFoundException var21) {
            check("ProcessGroupNativeWrapper loaded", false);
            System.out.println("  (re-run JavaCPP parse if missing)");
        }

        section("ProfilerResult / enableProfiler symbols");

        try {
            Class var23 = Class.forName("org.bytedeco.pytorch.profiler.ProfilerResult");
            check("ProfilerResult class", var23 != null);
            Method var18 = torch.class.getMethod("enableProfiler", ProfilerConfig.class, ActivityTypeSet.class);
            check("enableProfiler method", var18 != null);
            Method var19 = torch.class.getMethod("disableProfiler");
            check("disableProfiler method", var19 != null);
        } catch (Throwable var20) {
            check("profiler symbols", false);
            var20.printStackTrace(System.out);
        }

        System.out.println("\n==============================");
        System.out.println("Passed: " + passed + "  Failed: " + failed);
        System.out.println("==============================");
        if (failed > 0) {
            System.exit(1);
        }

    }
}
