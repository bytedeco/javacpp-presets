package dataframe;
//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//
import java.io.File;
import java.io.PrintStream;
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

public class MemoryProfilerExample {
    public MemoryProfilerExample() {
    }

    public static void main(String[] var0) {
        String var1 = "minimind_cpu_prof";
        BytePointer var2 = new BytePointer(var1);
        ExperimentalConfig var3 = new ExperimentalConfig();
        var3.verbose(true);
        var3.adjust_timestamps(true);
        var3.disable_external_correlation(true);
        ProfilerConfig var4 = new ProfilerConfig(ProfilerState.KINETO, true, true, true, true, true, var3, var2);
        ActivityTypeSet var5 = new ActivityTypeSet();
        var5.insert(ActivityType.CPU);
        System.out.println("Preparing Kineto profiler (enableProfiler path)...");
        torch.prepareProfiler(var4, var5);
        System.out.println("enableProfiler...");
        torch.enableProfiler(var4, var5);

        try {
            System.out.println("Running work (CPU matmul)...");
            TensorOptions var6 = (new TensorOptions()).dtype(new ScalarTypeOptional(ScalarType.Float));
            Tensor var7 = torch.randn(new long[]{2048L, 2048L}, var6);
            Tensor var8 = torch.randn(new long[]{2048L, 2048L}, var6);
            Tensor var9 = torch.mm(var7, var8);
            var9.add_(new Scalar((double)1.0F));
            var9.cpu().data_ptr();
        } catch (Exception var16) {
            var16.printStackTrace();
        } finally {
            System.out.println("disableProfiler + save...");
            ProfilerResult var11 = torch.disableProfiler();
            String var12 = "memory_profile.json";
            if (var11 != null && !var11.isNull()) {
                var11.save(var12);
            } else {
                System.err.println("disableProfiler returned null — no active session?");
            }

            File var13 = new File(var12);
            if (var13.exists() && var13.length() > 0L) {
                PrintStream var10000 = System.out;
                String var10001 = var13.getAbsolutePath();
                var10000.println("Export OK: " + var10001 + " (" + var13.length() + " bytes)");
            } else {
                System.err.println("Export failed or empty. Check that JavaCPP was regenerated");
                System.err.println("with enableProfiler/disableProfiler/ProfilerResult un-skipped,");
                System.err.println("and that libtorch was built with USE_KINETO (this Mac build has it).");
            }

            if (var11 != null) {
                var11.close();
            }

        }

    }
}
