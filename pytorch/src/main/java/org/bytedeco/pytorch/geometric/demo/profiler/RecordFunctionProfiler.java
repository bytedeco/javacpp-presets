package org.bytedeco.pytorch.geometric.demo.profiler;
import org.bytedeco.pytorch.profiler.*;
import org.bytedeco.pytorch.profiler.ExperimentalConfig;
import org.bytedeco.pytorch.profiler.ActivityTypeSet;
import org.bytedeco.pytorch.profiler.ProfilerConfig;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class RecordFunctionProfiler {
    public static void main(String[] args) {
        // 1. 基础配置：必须先开启 Profiler 才能让 RecordFunction 收集数据
        ProfilerConfig config = new ProfilerConfig(ProfilerState.KINETO);
        config.profile_memory(true);
        config.report_input_shapes(true);

        ActivityTypeSet activities = new ActivityTypeSet();
        activities.insert(ActivityType.CPU);

        System.out.println("🚀 启动底层 RecordFunction 监控...");
        prepareProfiler(config, activities);
        startMemoryProfile();

        try {
            // 2. 创建一个 RecordFunction 作用域
            // 注意：在 Java 中，RecordFunction 需要手动调用 end() 或者使用 try-with-resources
            try (RecordFunction guard = new RecordFunction(RecordScope.FUNCTION)) {

                // 模拟一个矩阵运算
                TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
                Tensor a = randn(new long[]{1024, 1024}, options);
                Tensor b = randn(new long[]{1024, 1024}, options);
                Tensor c = mm(a, b);
                c.cpu().data_ptr(); // 强制执行

                // 3. 检查 RecordFunction 是否捕捉到了信息
                if (guard.isActive()) {
                    System.out.println("--- 算子分析报告 ---");
                    System.out.println("算子名称: " + guard.name().getString());
                    System.out.println("输入数量: " + guard.num_inputs());

                    // 获取输入列表 (IValueArrayRef)
                    IValueArrayRef inputs = guard.inputs();
                    for (long i = 0; i < guard.num_inputs(); i++) {
                        IValue val = inputs.get(i);
                        if (val.isTensor()) {
                            Tensor t = val.toTensor();
                            long bytes = t.numel() * t.element_size();
                            System.out.println("  输入 [" + i + "] 形状: " + t.sizes().get(0) + "x" + t.sizes().get(1));
                            System.out.println("  输入 [" + i + "] 占用内存: " + (bytes / 1024) + " KB");
                        }
                    }
                } else {
                    System.out.println("⚠️ RecordFunction 未能捕捉到当前作用域的算子。");
                    System.out.println("提示：尝试将计算逻辑放入一个独立的子方法中调用。");
                }
            }
        } finally {
            stopMemoryProfile();
            System.out.println("🏁 监控关闭。");
        }
    }
}
