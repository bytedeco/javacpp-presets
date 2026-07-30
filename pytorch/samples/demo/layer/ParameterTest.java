package samples.demo.layer;
import org.bytedeco.pytorch.nn.*;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 最终版测试用例（修复打印格式，输出清晰）
 */
public class ParameterTest {
    public static void main(String[] args) {
        // 初始化环境
        torch.manual_seed(42);
        Device cpu = new Device(DeviceType.CPU);
        TensorOptions opts = new TensorOptions()
                .dtype(new ScalarTypeOptional(kFloat()))
                .device(new DeviceOptional(cpu))
                .requires_grad(new BoolOptional(true));

        // 1. 创建 Parameter
        Tensor data = randn(new long[]{2, 2}, opts);
        Parameter param = new Parameter(data, true);

        // 2. 手动设置梯度
        Tensor customGrad = ones(new long[]{2, 2}, opts);
        param.set_grad(customGrad);

        // 3. 验证梯度设置（修复打印格式）
        System.out.println("=== 手动设置的梯度 ===");
        printTensorFormatted(customGrad);

        System.out.println("\n=== Parameter 梯度 ===");
        printTensorFormatted(param.grad());

        // 4. 反向传播验证
        Tensor loss = param.sum();
        loss.backward();
        System.out.println("\n=== 反向传播后 Parameter 梯度 ===");
        printTensorFormatted(param.grad());

        // 5. 清空梯度测试
        param.zero_grad();
        System.out.println("\n=== 清空梯度后 ===");
        boolean isGradZero = (param.grad() == null) || isTensorEmpty(param.grad()) || (param.grad().sum().item_float() == 0);
        System.out.println("✅ 梯度是否为空/零: " + isGradZero);
    }

    /**
     * 修复版格式化打印（清晰展示张量）
     * @param tensor 要打印的张量
     */
    private static void printTensorFormatted(Tensor tensor) {
        if (tensor == null || isTensorEmpty(tensor)) {
            System.out.println("  [空张量]");
            return;
        }

        // 步骤1：获取张量基础信息
        long rows = tensor.size(0);
        long cols = tensor.size(1);
        ScalarType dtype = tensor.dtype().toScalarType();
        Device device = tensor.device();

        // 步骤2：安全获取张量数值（修复指针获取逻辑）
        float[] data = new float[(int) tensor.numel()];
        try {
            // 核心修复：bytedeco-pytorch 正确的浮点指针获取方式
            // 1. detach() 分离计算图 → 2. data_ptr() 获取数据指针 → 3. 转换为 FloatPointer → 4. 复制到数组
            FloatPointer floatPtr = new FloatPointer(tensor.detach().data_ptr());
            floatPtr.get(data); // 将指针数据拷贝到 float 数组
            floatPtr.close(); // 释放指针（避免内存泄漏）
        } catch (Exception e) {
            System.err.println("⚠️  读取张量数值失败: " + e.getMessage());
            return;
        }

        // 步骤3：模拟 Python 风格格式化打印
        System.out.println("  形状: (" + rows + ", " + cols + ")");
        System.out.println("  类型: " + dtype + " | 设备: " + device.is_cpu()+" "+ device.type().value+ device.toString());
        System.out.print("  数值: [");
        int idx = 0;
        for (long i = 0; i < rows; i++) {
            if (i > 0) System.out.print("        "); // 缩进对齐
            System.out.print("[");
            for (long j = 0; j < cols; j++) {
                System.out.printf("%.4f", data[idx++]);
                if (j < cols - 1) System.out.print(", ");
            }
            System.out.print("]");
            if (i < rows - 1) System.out.println(",");
        }
        System.out.println("]");
    }

    // 工具方法：判断张量是否为空
    private static boolean isTensorEmpty(Tensor tensor) {
        if (tensor == null) return true;
        try {
            return tensor.numel() == 0 ;
        } catch (Exception e) {
            return true;
        }
    }
}
//public class ParameterTest {
//    public static void main(String[] args) {
//        // 初始化环境
//        torch.manual_seed(42);
//        Device cpu = new Device(kCPU());
//        // 核心：必须开启 requires_grad，否则无梯度链路
//        TensorOptions opts = new TensorOptions()
//                .dtype(new ScalarTypeOptional(kFloat()))
//                .device(new DeviceOptional(cpu))
//                .requires_grad(new BoolOptional(true)); // 取消注释，开启梯度
//
//        // 1. 创建 Parameter（强制开启梯度）
//        Tensor data = randn(new long[]{2, 2}, opts);
//        Parameter param = new Parameter(data, true);
//
//        // 2. 手动设置梯度（测试 set_grad）
//        Tensor customGrad = ones(new long[]{2, 2}, opts);
//        param.set_grad(customGrad);
//
//        // 3. 验证梯度设置结果（修复 null 问题）
//        System.out.println("=== 手动设置的梯度 ===");
//        printTensor(customGrad);
//        System.out.println("\n=== Parameter 梯度 ===");
//        printTensor(param.grad());
//
//        // 4. 反向传播验证 hook 生效
//        Tensor loss = param.sum();
//        loss.backward(); // 触发 hook，生成底层梯度
//        System.out.println("\n=== 反向传播后 Parameter 梯度 ===");
//        printTensor(param.grad());
//
//        // 5. 清空梯度测试
//        param.zero_grad();
//        System.out.println("\n=== 清空梯度后 ===");
//        boolean isGradZero = (param.grad() == null) || (param.grad().sum().item_float() == 0);
//        System.out.println("梯度是否为空/零: " + isGradZero);
//    }
//
//    // 修复张量打印：适配 bytedeco-pytorch 原生 API
//    private static void printTensor(Tensor tensor) {
//        if (tensor == null) {
//            System.out.println("Tensor is null");
//            return;
//        }
//        // 原生打印方法，避免手动指针操作
//        torch.print(tensor);
//        // 可选：手动打印数值（兼容所有版本）
////        float[] data = new float[(int) tensor.numel()];
////        tensor.detach().data().asFloatPointer().get(data);
////        long rows = tensor.size(0);
////        long cols = tensor.size(1);
////        int idx = 0;
////        System.out.println("手动解析数值：");
////        for (long i = 0; i < rows; i++) {
////            for (long j = 0; j < cols; j++) {
////                System.out.printf("%.4f ", data[idx++]);
////            }
////            System.out.println();
////        }
//    }
//}
/**
 * 验证 set_grad() + register_hook 正确性
 */
//public class ParameterTest {
//    public static void main(String[] args) {
//        torch.manual_seed(42);
//        Device cpu = new Device(torch.kCPU());
//        TensorOptions opts = new TensorOptions()
//                .dtype(new ScalarTypeOptional(kFloat()))
//                .device(new DeviceOptional(cpu));
////                .requires_grad(new BoolOptional(true));
//
//        // 1. 创建 Parameter
//        Tensor data = randn(new long[]{2, 2}, opts);
//        Parameter param = new Parameter(data, true);
//
//        // 2. 手动设置梯度（测试 set_grad）
//        Tensor customGrad = ones(new long[]{2, 2}, opts);
//        param.set_grad(customGrad);
//
//        // 3. 验证梯度设置结果
//        System.out.println("手动设置的梯度:");
//        printTensor(customGrad);
//        System.out.println("\nParameter 梯度:");
//        printTensor(param.grad());
//
//        // 4. 反向传播验证 hook 生效
//        Tensor loss = param.sum();
//        loss.backward(); // 触发 hook
//
//        System.out.println("\n反向传播后 Parameter 梯度:");
//        printTensor(param.grad());
//
//        // 5. 清空梯度测试
//        param.zero_grad();
//        System.out.println("\n清空梯度后:");
//        System.out.println("梯度是否为 null: " + (param.grad() == null || param.grad().sum().item_float() == 0));
//    }
//
//    // 张量打印工具
//    private static void printTensor(Tensor tensor) {
//        torch.print(tensor);
////        float[] data = new float[(int) tensor.numel()];
////        tensor.detach().data().poi.get(data);
////        long rows = tensor.size(0);
////        long cols = tensor.size(1);
////        int idx = 0;
////        for (long i = 0; i < rows; i++) {
////            System.out.print("  ");
////            for (long j = 0; j < cols; j++) {
////                System.out.printf("%.4f ", data[idx++]);
////            }
////            System.out.println();
////        }
//    }
//}