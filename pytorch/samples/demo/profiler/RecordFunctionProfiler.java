package samples.demo.profiler;
import org.bytedeco.pytorch.profiler.*;

import org.bytedeco.pytorch.profiler.ActivityTypeSet;
import org.bytedeco.pytorch.profiler.ProfilerConfig;
import org.bytedeco.pytorch.profiler.ProfilerResult;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

import java.io.File;

/**
 * RecordFunction scopes only emit events when a Kineto session is active via
 * {@code enableProfiler} (not the Python-only memory profile APIs).
 */
public class RecordFunctionProfiler {
    public static void main(String[] args) {
        ProfilerConfig config = new ProfilerConfig(ProfilerState.KINETO);
        config.profile_memory(true);
        config.report_input_shapes(true);

        ActivityTypeSet activities = new ActivityTypeSet();
        activities.insert(ActivityType.CPU);

        System.out.println("Starting RecordFunction + Kineto profiler...");
        prepareProfiler(config, activities);
        enableProfiler(config, activities);

        try {
            try (RecordFunction guard = new RecordFunction(RecordScope.FUNCTION)) {
                TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
                Tensor a = randn(new long[]{1024, 1024}, options);
                Tensor b = randn(new long[]{1024, 1024}, options);
                Tensor c = mm(a, b);
                c.cpu().data_ptr();

                if (guard.isActive()) {
                    System.out.println("--- RecordFunction report ---");
                    System.out.println("name: " + guard.name().getString());
                    System.out.println("num_inputs: " + guard.num_inputs());
                    IValueArrayRef inputs = guard.inputs();
                    for (long i = 0; i < guard.num_inputs(); i++) {
                        IValue val = inputs.get(i);
                        if (val.isTensor()) {
                            Tensor t = val.toTensor();
                            long bytes = t.numel() * t.element_size();
                            System.out.println("  input[" + i + "] numel=" + t.numel()
                                    + " bytes=" + (bytes / 1024) + " KB");
                        }
                    }
                } else {
                    System.out.println("RecordFunction not active in this scope.");
                }
            }
        } finally {
            ProfilerResult result = disableProfiler();
            String path = "record_function_profile.json";
            if (result != null && !result.isNull()) {
                result.save(path);
                result.close();
            }
            File f = new File(path);
            System.out.println(f.exists() && f.length() > 0
                    ? "Saved " + f.getAbsolutePath() + " (" + f.length() + " bytes)"
                    : "No trace file — regenerate JavaCPP with ProfilerResult un-skipped.");
        }
    }
}
