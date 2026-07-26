package org.bytedeco.pytorch.geometric.demo.layer;

//package org.bytedeco.pytorch.geometric.demo;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GENConv;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * GENConv 完整测试用例（覆盖 Softmax/PowerMean 聚合、边特征、反向传播）
 */
public class GENConvTest {
    public static void main(String[] args) {
        // 1. 初始化环境
        torch.manual_seed(42); // 固定随机种子
        Device cpu = new Device(kCPU());
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(kFloat()))
                .device(new DeviceOptional(cpu))
                .requires_grad(new BoolOptional(true));
        TensorOptions longOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(kLong()))
                .device(new DeviceOptional(cpu));

        // 2. 测试参数
        long inChannels = 6;     // 输入通道
        long outChannels = 8;    // 输出通道
        long numNodes = 10;      // 节点数
        long numEdges = 15;      // 边数
        int edgeDim = 4;        // 边特征维度

        // 3. 生成测试数据
        // 节点特征 [N, C]
        Tensor x = randn(new long[]{numNodes, inChannels}, floatOpts);
        // 边索引 [2, E]
        long[] edgeData = {
                0,1, 0,2, 1,3, 2,3, 3,4, 4,5, 5,6, 6,7, 7,8, 8,9,
                1,2, 2,4, 4,6, 6,8, 8,0
        };
        Tensor edgeIndex = tensor(edgeData, longOpts).view(2, numEdges);
        // 边特征 [E, edgeDim]
        Tensor edgeAttr = randn(new long[]{numEdges, edgeDim}, floatOpts);

        // 4. 测试 1: Softmax 聚合（GENConv 标准）
        System.out.println("=== 测试 1: Softmax 聚合 ===");
        GENConv genConvSoftmax = new GENConv(inChannels, outChannels,
                "softmax", 1.0f, false, 1.0f, false, edgeDim, 1e-7f, true);
        Tensor outSoftmax = genConvSoftmax.forward(x, edgeIndex, edgeAttr);
        validateOutput(outSoftmax, numNodes, outChannels, "Softmax");

        // 5. 测试 2: PowerMean 聚合
        System.out.println("\n=== 测试 2: PowerMean 聚合 ===");
        GENConv genConvPowerMean = new GENConv(inChannels, outChannels,
                "powermean", 1.0f, false, 2.0f, false, edgeDim, 1e-7f, true);
        Tensor outPowerMean = genConvPowerMean.forward(x, edgeIndex, edgeAttr);
        validateOutput(outPowerMean, numNodes, outChannels, "PowerMean");

        // 6. 测试 3: 无边形特征输入
        System.out.println("\n=== 测试 3: 无边形特征 ===");
        Tensor outNoEdge = genConvSoftmax.forward(x, edgeIndex);
        validateOutput(outNoEdge, numNodes, outChannels, "无边形特征");

        // 7. 测试 4: 反向传播
        System.out.println("\n=== 测试 4: 反向传播 ===");
        Tensor loss = outSoftmax.sum();
        loss.backward(); // 反向传播
        System.out.println("✅ 反向传播完成，无异常！");

        // 8. 测试 5: 学习聚合参数（learnT/learnP）
        System.out.println("\n=== 测试 5: 可学习聚合参数 ===");
        GENConv genConvLearn = new GENConv(inChannels, outChannels,
                "softmax", 1.0f, true, 2.0f, true, edgeDim, 1e-7f, true);
        Tensor outLearn = genConvLearn.forward(x, edgeIndex, edgeAttr);
        System.out.println("可学习 t 参数 requires_grad: " + genConvLearn.getT().requires_grad()); // true
        System.out.println("可学习 p 参数 requires_grad: " + genConvLearn.getP().requires_grad()); // true
        validateOutput(outLearn, numNodes, outChannels, "可学习参数");

        // 9. 释放资源
        x.close();
        edgeIndex.close();
        edgeAttr.close();
        outSoftmax.close();
        outPowerMean.close();
        outNoEdge.close();
        loss.close();
        outLearn.close();
        genConvSoftmax.close();
        genConvPowerMean.close();
        genConvLearn.close();

        System.out.println("\n🎉 所有测试通过！");
    }

    // 输出验证工具
    private static void validateOutput(Tensor out, long expectedNodes, long expectedChannels, String testName) {
        // 维度验证
        if (out.size(0) != expectedNodes || out.size(1) != expectedChannels) {
            throw new RuntimeException(testName + " 维度验证失败: 输出=" + out.size(0) + "x" + out.size(1) +
                    ", 预期=" + expectedNodes + "x" + expectedChannels);
        }
        System.out.println(testName + " 维度验证通过: " + out.size(0) + "x" + out.size(1));

        // 数值异常验证
        boolean hasNaN = out.isnan().any().item().toBool();
        boolean hasInf = out.isinf().any().item().toBool();
        if (hasNaN || hasInf) {
            throw new RuntimeException(testName + " 包含数值异常: NaN=" + hasNaN + ", Inf=" + hasInf);
        }
        System.out.println(testName + " 数值异常验证通过（无 NaN/Inf）");

        // 打印前3行输出（直观验证）
        System.out.println(testName + " 输出前3行:");
        printTensor(out.slice(0,new LongOptional( 0), new LongOptional(3),1));
    }

    // 张量打印工具
    private static void printTensor(Tensor tensor) {
        float[] data = new float[(int) tensor.numel()];
        tensor.detach().data_ptr_float().get(data);
        int cols = (int) tensor.size(1);
        for (int i = 0; i < tensor.size(0); i++) {
            System.out.print("  ");
            for (int j = 0; j < Math.min(cols, 6); j++) { // 只打印前6列
                System.out.printf("%.4f ", data[i * cols + j]);
            }
            if (cols > 6) System.out.print("...");
            System.out.println();
        }
    }
}
