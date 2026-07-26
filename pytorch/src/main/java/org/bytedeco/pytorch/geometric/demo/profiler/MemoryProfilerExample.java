package org.bytedeco.pytorch.geometric.demo.profiler;
import org.bytedeco.pytorch.profiler.*;
import org.bytedeco.pytorch.profiler.ExperimentalConfig;
import org.bytedeco.pytorch.profiler.ActivityTypeSet;
import org.bytedeco.pytorch.profiler.ProfilerConfig;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

import java.io.File;

public class MemoryProfilerExample {
    public static void main(String[] args) {
        // 1. 拟定一个明确的 traceId
        String traceIdStr = "minimind_cpu_prof";
        BytePointer traceId = new BytePointer(traceIdStr);
        // 2. 配置实验性参数：在 Mac 上禁用外部关联非常重要
        ExperimentalConfig experimentalConfig = new ExperimentalConfig();
        experimentalConfig.verbose(true);
        experimentalConfig.adjust_timestamps(true); // 尝试对齐 Mac 系统的事件时间戳
        experimentalConfig.disable_external_correlation(true); // 禁用外部（CUDA等）关联
        // 3. 修改 ProfilerState 为 CPU (避免 Kineto 的多设备切换告警)
        ProfilerConfig config = new ProfilerConfig(
                ProfilerState.KINETO, // 改用 CPU 模式，更适合 Mac 推理分析
                true,  // report_input_shapes
                true,  // profile_memory
                true,  // with_stack
                true,  // with_flops
                true,  // with_modules
                experimentalConfig,
                traceId
        );

        ActivityTypeSet activities = new ActivityTypeSet();
        activities.insert(ActivityType.CPU);

        System.out.println("正在准备分析器 (CPU Mode)...");
        prepareProfiler(config, activities);

        // 4. 开启记录
        startMemoryProfile();

        try {
            System.out.println("执行模型推理 (CPU Matrix Mul)...");

            // 进一步加大计算量，确保 native 缓冲区有足够数据
            TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
            Tensor input = randn(new long[]{5000, 5000}, options);
            Tensor weight = randn(new long[]{5000, 5000}, options);

            // 执行矩阵乘法并强制执行
            Tensor output = mm(input, weight);
            output.add_(new Scalar(1.0)); // 额外操作确保算子链完整
            output.cpu().data_ptr(); // 强制同步

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            System.out.println("正在停止分析器并刷新数据...");
            stopMemoryProfile();

            // 5. 导出结果
            // 使用当前项目路径下的简单名称
            String exportPath = "memory_profile.json";
            exportMemoryProfile(exportPath);

            File file = new File(exportPath);
            if (file.exists() && file.length() > 0) {
                System.out.println("✅ 导出成功！");
                System.out.println("文件路径: " + file.getAbsolutePath());
                System.out.println("文件大小: " + file.length() + " bytes");
            } else {
                System.err.println("❌ 仍然导出失败。这可能是因为当前 JavaCPP 绑定的 LibTorch 裁剪了 Kineto 导出组件。");
                System.err.println("建议：检查项目根目录下是否有名为 'trace.json' 的默认生成文件。");
            }
        }
    }
}
//public class MemoryProfilerExample {
//    public static void main(String[] args) {
//        // 1. 拟定一个 traceId 和配置实验性参数
//        String traceIdStr = "minimind_inference_run_1";
//        BytePointer traceId = new BytePointer(traceIdStr);
//        // 构造 ExperimentalConfig 并设置关键参数
//        ExperimentalConfig experimentalConfig = new ExperimentalConfig();
//        experimentalConfig.verbose(true); // 开启详细模式，有助于强制数据写盘
//        experimentalConfig.profile_all_threads(true); // 捕获所有线程的内存分配
//        experimentalConfig.disable_external_correlation(true); // 关键：Mac 没 CUDA，禁用它
//        // 2. 初始化配置
//        ProfilerConfig config = new ProfilerConfig(
//                ProfilerState.CPU,
//                true,  // report_input_shapes
//                true,  // profile_memory
//                true,  // with_stack
//                true,  // with_flops
//                true,  // with_modules
//                experimentalConfig,
//                traceId
//        );
//
//        // 3. 设置活动类型
//        ActivityTypeSet activities = new ActivityTypeSet();
//        activities.insert(ActivityType.CPU);
//
//        System.out.println("正在准备分析器...");
//        prepareProfiler(config, activities);
//
//        // 4. 开始内存记录
//        startMemoryProfile();
//        // 显式开启 Kineto 收集
//        toggleCollectionDynamic(true, activities);
//
//        try {
//            System.out.println("执行模型推理 (CPU)...");
//
//            // 增大计算量，确保 Profiler 有足够的时间捕捉到事件
//            TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
//            Tensor input = randn(new long[]{4096, 4096}, options);
//            Tensor weight = randn(new long[]{4096, 4096}, options);
//
//            // 执行矩阵乘法
//            Tensor output = mm(input, weight);
//
//            // 强制同步：确保计算真正完成
//            output.cpu().data_ptr();
//            System.out.println("计算完成，正在停止分析器...");
//
//        } catch (Exception e) {
//            e.printStackTrace();
//        } finally {
//            // 5. 停止并强制刷新缓冲区
//            toggleCollectionDynamic(false, activities);
//            stopMemoryProfile();
//
//            // 6. 导出分析结果
//            // 先导出到当前工作目录的简单文件名
//            String fileName = "memory_profile.json";
//            exportMemoryProfile(fileName);
//
//            File exportedFile = new File(fileName);
//            if (exportedFile.exists()) {
//                System.out.println("✅ 导出成功！文件大小: " + exportedFile.length() + " bytes");
//                System.out.println("文件绝对路径: " + exportedFile.getAbsolutePath());
//            } else {
//                System.err.println("❌ 导出失败：文件未生成。请检查是否有磁盘写入权限。");
//            }
//        }
//    }
//}
//public class MemoryProfilerExample {
//    public static void main(String[] args) {
//        // 1. 显式创建 ExperimentalConfig 和 TraceID，确保不是悬空指针
//        ExperimentalConfig experimentalConfig = new ExperimentalConfig();
//        // trace_id 传入一个明确的空字符串，而不是未初始化的 BytePointer
//        BytePointer traceId = new BytePointer("");
//
//        // 2. 初始化配置
//        // 确保 ProfilerState 使用 KINETO，这是导出 JSON 必须的状态
//        ProfilerConfig config = new ProfilerConfig(
//                ProfilerState.KINETO, // 状态
//                true,  // report_input_shapes
//                true,  // profile_memory (显存/内存记录核心)
//                true,  // with_stack
//                true,  // with_flops (Mac CPU 建议开启)
//                true,  // with_modules
//                experimentalConfig,
//                traceId
//        );
//
//        // 3. 设置活动类型
//        ActivityTypeSet activities = new ActivityTypeSet();
//        activities.insert(ActivityType.CPU);
//
//        // 注意：在 Mac 上如果使用了 MPS，LibTorch 有时会将事件记录在 CPU 活动流中，
//        // 或者需要特定版本的支持。目前先确保 CPU 记录完整。
//
//        System.out.println("正在准备分析器...");
//        // 这一步非常重要，它初始化了底层的事件监听器
//        prepareProfiler(config, activities);
//
//        // 4. 开启内存分析
//        startMemoryProfile();
//
//        // 补充：在某些版本中，除了 startMemoryProfile，
//        // 可能还需要 toggleCollectionDynamic 来确保事件被推送到 Kineto 队列
//        toggleCollectionDynamic(true, activities);
//
//        try {
//            System.out.println("执行模型推理 (CPU)...");
//
//            // 使用标准的 TensorOptions 创建方式
//            TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
//
//            Tensor input = randn(new long[]{2048, 2048}, options);
//            Tensor weight = randn(new long[]{2048, 2048}, options);
//
//            // 执行矩阵乘法
//            Tensor output = mm(input, weight);
//
//            // 确保计算实际发生（由于 LibTorch 的延迟执行特性）
//            output.cpu().data_ptr();
//
//        } catch (Exception e) {
//            e.printStackTrace();
//        } finally {
//            // 5. 停止记录
//            toggleCollectionDynamic(false, activities);
//            stopMemoryProfile();
//
//            // 6. 导出分析结果
//            // 建议先使用相对路径或确保目录存在
//            String exportPath = "memory_profile.json";
//            System.out.println("分析完成，结果导出至: " + new java.io.File(exportPath).getAbsolutePath());
//
//            // 核心调用
//            exportMemoryProfile(exportPath);
//        }
//
//        if (profilerEnabled()) {
//            System.out.println("分析器已关闭。");
//        }
//    }
//}
//public class MemoryProfilerExample {
//    public static void main(String[] args) {
//        // 1. 初始化配置
//        // ProfilerConfig 参数通常包括: 
//        // state, report_input_shapes, profile_memory, with_stack, with_flops, with_modules
//        // 这里通过构造函数启用内存分析 (profile_memory = true)
//        ProfilerConfig config = new ProfilerConfig(
//                ProfilerState.KINETO, // 状态
//                true,  // report_input_shapes
//                true,  // profile_memory (关键：记录显存)
//                true,  // with_stack
//                false, // with_flops
//                false,  // with_modules
//                new ExperimentalConfig(),
//                new BytePointer()
//        );
//
//        // 2. 设置活动类型 (CPU, CUDA 等)
//        // 假设我们在 GPU 上运行，需要记录 CUDA 显存
//        ActivityTypeSet activities = new ActivityTypeSet();
//        activities.insert(ActivityType.CPU);
////        activities.insert(ActivityType.CUDA);
//
//        System.out.println("正在准备分析器...");
//        prepareProfiler(config, activities);
//
//        // 3. 开始内存记录 (调用 api 中的 startMemoryProfile)
//        startMemoryProfile();
//
//        try {
//            // --- 模拟推理过程 ---
//            System.out.println("执行模型推理...");
//            Tensor input = randn(new long[]{1024, 1024},new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));//.cuda();
//            Tensor weight = randn(new long[]{1024, 1024},new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));//.cuda();
//            Tensor output = mm(input, weight);
//            // ------------------
//        } finally {
//            // 4. 停止内存记录
//            stopMemoryProfile();
//
//            // 5. 导出分析结果 (支持导出为 JSON 格式，可用 Chrome tracing 查看)
//            String exportPath = "/Users/mullerzhang/IdeaProjects/torch-geometric/src/main/java/torch/geometric/demo/memory_profile.json";
//            System.out.println("分析完成，结果导出至: " + exportPath);
//            exportMemoryProfile(exportPath);
//        }
//
//        // 检查 Profiler 是否处于启用状态
//        if (profilerEnabled()) {
//            System.out.println("当前分析器类型: " + profilerType());
//        }
//    }
//}