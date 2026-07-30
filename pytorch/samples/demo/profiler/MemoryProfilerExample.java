package samples.demo.profiler;
import org.bytedeco.pytorch.profiler.*;

import org.bytedeco.pytorch.profiler.ActivityTypeSet;
import org.bytedeco.pytorch.profiler.ExperimentalConfig;
import org.bytedeco.pytorch.profiler.ProfilerConfig;
import org.bytedeco.pytorch.profiler.ProfilerResult;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

import java.io.File;

/**
 * CPU (and optionally CUDA) Kineto profiler demo that actually writes a
 * chrome-trace JSON via {@code enableProfiler} / {@code disableProfiler} /
 * {@link ProfilerResult#save(String)}.
 *
 * <p><b>Do not</b> use {@code startMemoryProfile}/{@code exportMemoryProfile}
 * from pure JavaCPP: those APIs only work when a Python memory tracer is
 * registered ({@code torch.cuda.memory}); without Python they are NoOps and
 * never create a file.
 *
 * <p>On Mac (no CUDA): {@code ProfilerState.KINETO} + {@code ActivityType.CPU}
 * + {@code profile_memory=true} records CPU allocator events into the Kineto
 * trace. True CUDA device-memory snapshots need a Linux-GPU libtorch build.
 */
public class MemoryProfilerExample {
    public static void main(String[] args) {
        String traceIdStr = "minimind_cpu_prof";
        BytePointer traceId = new BytePointer(traceIdStr);

        ExperimentalConfig experimentalConfig = new ExperimentalConfig();
        experimentalConfig.verbose(true);
        experimentalConfig.adjust_timestamps(true);
        // Mac has no CUPTI/CUDA external correlation.
        experimentalConfig.disable_external_correlation(true);

        ProfilerConfig config = new ProfilerConfig(
                ProfilerState.KINETO,
                true,  // report_input_shapes
                true,  // profile_memory  — CPU allocator events on Mac
                true,  // with_stack
                true,  // with_flops
                true,  // with_modules
                experimentalConfig,
                traceId
        );

        ActivityTypeSet activities = new ActivityTypeSet();
        activities.insert(ActivityType.CPU);
        // When running against a CUDA-enabled libtorch, also:
        // activities.insert(ActivityType.CUDA);

        System.out.println("Preparing Kineto profiler (enableProfiler path)...");
        prepareProfiler(config, activities);

        System.out.println("enableProfiler...");
        enableProfiler(config, activities);

        try {
            System.out.println("Running work (CPU matmul)...");
            TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
            Tensor input = randn(new long[]{2048, 2048}, options);
            Tensor weight = randn(new long[]{2048, 2048}, options);
            Tensor output = mm(input, weight);
            output.add_(new Scalar(1.0));
            // Force materialization / sync.
            output.cpu().data_ptr();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            System.out.println("disableProfiler + save...");
            ProfilerResult result = disableProfiler();
            String exportPath = "memory_profile.json";
            if (result != null && !result.isNull()) {
                result.save(exportPath);
            } else {
                System.err.println("disableProfiler returned null — no active session?");
            }

            File file = new File(exportPath);
            if (file.exists() && file.length() > 0) {
                System.out.println("Export OK: " + file.getAbsolutePath()
                        + " (" + file.length() + " bytes)");
            } else {
                System.err.println("Export failed or empty. Check that JavaCPP was regenerated");
                System.err.println("with enableProfiler/disableProfiler/ProfilerResult un-skipped,");
                System.err.println("and that libtorch was built with USE_KINETO (this Mac build has it).");
            }
            if (result != null) {
                result.close();
            }
        }
    }
}
