package org.bytedeco.pytorch.geometric.demo.profiler;
import org.bytedeco.pytorch.profiler.*;
import org.bytedeco.pytorch.profiler.ExperimentalConfig;
import org.bytedeco.pytorch.profiler.ActivityTypeSet;
import org.bytedeco.pytorch.profiler.ProfilerConfig;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class ManualMemoryProfiler {
    public static void main(String[] args) {
        // 1. 初始化基础配置
        ProfilerConfig config = new ProfilerConfig(ProfilerState.KINETO);
        config.profile_memory(true);
        config.report_input_shapes(true);

        ActivityTypeSet activities = new ActivityTypeSet();
        activities.insert(ActivityType.CPU);

        System.out.println("🚀 开启手动内存监控...");
        prepareProfiler(config, activities);
        startMemoryProfile();

        try {
            // 使用 RecordFunction 手动标记一个要观察的代码块
            // 这会触发底层显存记录
            try (RecordFunction guard = new RecordFunction(RecordScope.USER_SCOPE)) {
//                guard.before("MiniMind_Forward_Step");
                System.out.println("执行推理中...");
                TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
                Tensor input = randn(new long[]{2048, 2048}, options);
                Tensor weight = randn(new long[]{2048, 2048}, options);
                Tensor output = mm(input, weight);

                // 关键点：使用你提供的 API 获取输入大小信息
                // 注意：RecordFunction 会记录当前作用域内的算子
//                LongVector sizes = inputSizes(guard, true);
//                if (!sizes.empty()) {
//                    System.out.println("检测到算子输入维度: " + sizes.get(0));
//                }

                output.cpu().data_ptr();
                System.out.println("推理完成。");
            }
        } finally {
            stopMemoryProfile();
            System.out.println("✅ 监控结束。");

            // 如果 export 还是失败，我们至少可以通过 Java 侧获取当前进程显存
            // 在 Mac MPS 模式下，可以使用以下逻辑观察内存波动
            long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            System.out.println("当前 JVM 预估占用内存: " + (usedMemory / 1024 / 1024) + " MB");
        }
    }
}