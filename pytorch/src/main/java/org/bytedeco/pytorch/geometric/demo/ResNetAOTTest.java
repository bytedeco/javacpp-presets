package org.bytedeco.pytorch.geometric.demo;
import org.bytedeco.pytorch.inductor.*;
import org.bytedeco.pytorch.inductor.AOTIModelContainerRunnerCpu;

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class ResNetAOTTest {
    public static void main(String[] args) {
        // 1. 模型路径
        System.setProperty("org.bytedeco.openblas.load", "nolapack");
        String modelPath ="/Users/mullerzhang/Documents/code/langchain/resnet18_aot/data/aotinductor/model/cgoreo2mzfcktrewthdqntfumlvwuom6p7gtmb2jj4vrr73kcllj.wrapper.so"; // parse from pt2 zip
        // "/Users/mullerzhang/Documents/code/langchain/model/data/aotinductor/model/cf4ya7bndiovsxlxek5xbriu3622dvdameazkarl5gcajcdtwki7.wrapper.so"; 
        // "/Users/mullerzhang/Documents/code/langchain/model.pt2";// "/Users/mullerzhang/Documents/code/langchain/resnet18_aot.pt2";

        // 2. 初始化 AOTI 运行器
        // AOTIModelContainerRunnerCpu 构造函数：路径, 模型实例数, 是否单线程
        try (AOTIModelContainerRunnerCpu runner = new AOTIModelContainerRunnerCpu(modelPath, 1, true)) {

            // 3. 构造输入 Tensor (必须与导出时的 Shape 一致: 1, 3, 224, 224)
            long[] shape = {1, 3, 224, 224};
//            long[] shape = {8,10};
            Tensor inputTensor = randn(shape, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(new Device(DeviceType.CPU)))).contiguous();

            // 4. 将输入放入 TensorVector
            TensorVector inputs = new TensorVector(inputTensor);

            System.out.println("开始 AOT 推理...");
            long startTime = System.currentTimeMillis();

            // 5. 执行推理
            TensorVector outputs = runner.run(inputs);

            long endTime = System.currentTimeMillis();
            System.out.println("推理耗时: " + (endTime - startTime) + "ms");

            // 6. 检查输出 (ResNet18 输出通常是 [1, 1000])
            Tensor outputTensor = outputs.get(0);
            System.out.println("输出形状: " + java.util.Arrays.toString(outputTensor.sizes().vec().get()));

            // 打印前 5 个类别的置信度
            FloatPointer data = outputTensor.data_ptr_float();
            System.out.print("前 5 个输出值: ");
            for (int i = 0; i < 5; i++) {
                System.out.print(data.get(i) + " ");
            }
            System.out.println();

        } catch (Exception e) {
            System.err.println("AOT 加载或推理失败，请检查 .so 与当前环境的兼容性");
            e.printStackTrace();
        }
    }
}