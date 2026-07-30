package samples.demo.layer;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GINEConv;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * GINEConv 完整测试用例（覆盖有/无边特征、可学习/不可学习 eps、反向传播）
 */
public class GINEConvTest {
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
        long nodeDim = 6;       // 节点特征维度
        long edgeDim = 4;       // 边特征维度（与节点维度不一致，测试对齐）
        long outDim = 8;        // MLP 输出维度
        long numNodes = 10;     // 节点数
        long numEdges = 15;     // 边数

        // 3. 构建核心 MLP（GINEConv 标配）
        SequentialImpl nn = new SequentialImpl();
        var options = new LinearOptions(nodeDim, outDim);
        options.bias().put(true);
        nn.push_back(new LinearImpl(options));
        nn.push_back(new ReLUImpl());
        var options2 = new LinearOptions(outDim, outDim);
        options2.bias().put(true);
        nn.push_back(new LinearImpl(options2));

        // 4. 生成测试数据
        // 节点特征 [N, C]
        Tensor x = randn(new long[]{numNodes, nodeDim}, floatOpts);
        // 边索引 [2, E]
        long[] edgeData = {
                0,1, 0,2, 1,3, 2,3, 3,4, 4,5, 5,6, 6,7, 7,8, 8,9,
                1,2, 2,4, 4,6, 6,8, 8,0
        };
        Tensor edgeIndex = tensor(edgeData, longOpts).view(2, numEdges);
        // 边特征 [E, edgeDim]
        Tensor edgeAttr = randn(new long[]{numEdges, edgeDim}, floatOpts);

        // 5. 测试 1: 不可学习 eps + 有边特征
        System.out.println("=== 测试 1: 不可学习 eps + 有边特征 ===");
        GINEConv gineConv1 = new GINEConv(nn, 0.0, false, (int) edgeDim, (int) nodeDim);
        Tensor out1 = gineConv1.forward(x, edgeIndex, edgeAttr);
        validateOutput(out1, numNodes, outDim, "不可学习 eps + 有边特征");

        // 6. 测试 2: 可学习 eps + 无边形特征（退化为 GIN）
        System.out.println("\n=== 测试 2: 可学习 eps + 无边形特征 ===");
        GINEConv gineConv2 = new GINEConv(nn, 0.1, true, null, (int) nodeDim);
        Tensor out2 = gineConv2.forward(x, edgeIndex);
        validateOutput(out2, numNodes, outDim, "可学习 eps + 无边形特征");

        // 7. 测试 3: 无 MLP 场景（仅聚合）
        System.out.println("\n=== 测试 3: 无 MLP + 有边特征 ===");
        GINEConv gineConv3 = new GINEConv(null, 0.0, false, (int) edgeDim, (int) nodeDim);
        Tensor out3 = gineConv3.forward(x, edgeIndex, edgeAttr);
        validateOutput(out3, numNodes, nodeDim, "无 MLP + 有边特征"); // 输出维度为节点维度

        // 8. 测试 4: 反向传播
        System.out.println("\n=== 测试 4: 反向传播 ===");
        Tensor loss = out1.sum();
        loss.backward(); // 反向传播
        System.out.println("✅ 反向传播完成，无异常！");
        // 验证可学习参数梯度
//        if (gineConv2.epsParam.grad() != null) {
//            System.out.println("可学习 eps 梯度: " + gineConv2.epsParam.grad().item_float());
//        }
        if (nn != null && !nn.is_empty()) {
            var  linear = nn.get(0).asLinear();
            if (linear.weight().grad() != null && linear.weight().grad().numel() > 0) {
                System.out.println("MLP 第一层权重梯度和: " + linear.weight().grad().sum().item_float());
            }

        }

        // 9. 释放资源
        x.close();
        edgeIndex.close();
        edgeAttr.close();
        out1.close();
        out2.close();
        out3.close();
        loss.close();
        gineConv1.close();
        gineConv2.close();
        gineConv3.close();
        nn.close();

        System.out.println("\n🎉 所有测试通过！");
    }

    // 输出验证工具
    private static void validateOutput(Tensor out, long expectedNodes, long expectedChannels, String testName) {
        // 维度验证
        if (out.size(0) != expectedNodes || out.size(1) != expectedChannels) {
            throw new RuntimeException(
                    testName + " 维度验证失败: 输出=" + out.size(0) + "x" + out.size(1) +
                            ", 预期=" + expectedNodes + "x" + expectedChannels
            );
        }
        System.out.println(testName + " 维度验证通过: " + out.size(0) + "x" + out.size(1));

        // 数值异常验证
        boolean hasNaN = out.isnan().any().item().toBool();
        boolean hasInf = out.isinf().any().item().toBool();
        if (hasNaN || hasInf) {
            throw new RuntimeException(testName + " 包含数值异常: NaN=" + hasNaN + ", Inf=" + hasInf);
        }
        System.out.println(testName + " 数值异常验证通过（无 NaN/Inf）");

        // 打印前3行输出
        System.out.println(testName + " 输出前3行:");
        printTensor(out.slice(0, new LongOptional(0), new LongOptional(3),1));
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