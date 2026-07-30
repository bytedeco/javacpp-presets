package samples.demo.profiler;
import org.bytedeco.pytorch.profiler.*;

import org.bytedeco.pytorch.profiler.ActivityTypeSet;
import org.bytedeco.pytorch.profiler.ProfilerConfig;
import org.bytedeco.pytorch.profiler.ProfilerResult;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

import java.io.File;

/**
 * Minimal Kineto profiler + optional {@link RecordFunction} user scope.
 *
 * <p>Uses the real chrome-trace path ({@code enableProfiler} /
 * {@code disableProfiler} / {@link ProfilerResult#save}). The
 * {@code startMemoryProfile}/{@code exportMemoryProfile} pair is Python-only
 * and is a NoOp under pure JavaCPP.
 */
public class ManualMemoryProfiler {
    public static void main(String[] args) {
        ProfilerConfig config = new ProfilerConfig(ProfilerState.KINETO);
        config.profile_memory(true);
        config.report_input_shapes(true);

        ActivityTypeSet activities = new ActivityTypeSet();
        activities.insert(ActivityType.CPU);

        System.out.println("Preparing profiler...");
        prepareProfiler(config, activities);
        enableProfiler(config, activities);

        try {
            try (RecordFunction guard = new RecordFunction(RecordScope.USER_SCOPE)) {
                System.out.println("Running inference block...");
                TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
                Tensor input = randn(new long[]{1024, 1024}, options);
                Tensor weight = randn(new long[]{1024, 1024}, options);
                Tensor output = mm(input, weight);
                output.cpu().data_ptr();
                System.out.println("Done.");
            }
        } finally {
            ProfilerResult result = disableProfiler();
            String path = "manual_memory_profile.json";
            if (result != null && !result.isNull()) {
                result.save(path);
                result.close();
            }
            File f = new File(path);
            if (f.exists() && f.length() > 0) {
                System.out.println("Saved " + f.getAbsolutePath() + " (" + f.length() + " bytes)");
            } else {
                System.err.println("No trace written — regenerate JavaCPP with ProfilerResult un-skipped.");
            }
            long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            System.out.println("JVM heap estimate: " + (usedMemory / 1024 / 1024) + " MB");
        }
    }
}
