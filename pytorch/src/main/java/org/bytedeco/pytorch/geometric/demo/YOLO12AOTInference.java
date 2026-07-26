package org.bytedeco.pytorch.geometric.demo;
import org.bytedeco.pytorch.inductor.*;
import org.bytedeco.pytorch.inductor.AOTIModelContainerRunnerCpu;

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

import java.util.Arrays;

public class YOLO12AOTInference {
    public static void main(String[] args) {
        // 1. 指定 .so 模型路径 (确保路径正确)
        String modelPath = "/Users/mullerzhang/Documents/code/langchain/yolo12_aot/data/aotinductor/model/cth7gpzhojzjn3pwhhdttlp7u5snht65f362t6lkxobt45qtdgjh.wrapper.so";

        // 2. 环境准备：强制单线程以避免 macOS 上的 OpenMP 冲突 (SIGSEGV)
        System.setProperty("org.bytedeco.openblas.load", "nolapack");

        // 3. 使用 AOTIModelContainerRunnerCpu 加载模型
        // 参数: 路径, 实例数=1, 强制单线程=true
        try (AOTIModelContainerRunnerCpu runner = new AOTIModelContainerRunnerCpu(modelPath, 1, true)) {

            System.out.println("✅ AOT 模型加载成功!");

            // 4. 构造输入 Tensor (Shape: 1, 3, 640, 640)
            // 注意：dtype 必须匹配导出时的 Float32
            long[] inputShape = {1, 3, 640, 640};
            Tensor inputTensor = randn(inputShape, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).contiguous();

            // 5. 封装输入为 TensorVector (AOTI 接收 Vector 容器)
            TensorVector inputs = new TensorVector(inputTensor);

            System.out.println("🚀 开始推理...");
            long startTime = System.nanoTime();

            // 6. 执行推理
            // runner.run 会返回一个 TensorVector，包含模型所有的输出节点
            TensorVector outputs = runner.run(inputs);

            long endTime = System.nanoTime();
            System.out.printf("⏱️ 推理耗时: %.2f ms%n", (endTime - startTime) / 1e6);

            // 7. 处理输出结果
            System.out.println("📦 检测到输出层数量: " + outputs.size());

            for (int i = 0; i < outputs.size(); i++) {
                Tensor out = outputs.get(i);
                long[] outShape = out.sizes().vec().get();
                System.out.println("层 " + i + " 形状: " + Arrays.toString(outShape));

                // 如果你想读取具体数值 (例如前 10 个)
                FloatPointer data = out.data_ptr_float();
                System.out.print("前 10 个原始数值: ");
                for (int j = 0; j < Math.min(10, out.numel()); j++) {
                    System.out.print(data.get(j) + " ");
                }
                System.out.println("\n---");
            }

        } catch (Exception e) {
            System.err.println("❌ AOT 推理失败:");
            e.printStackTrace();
        }
    }
}