package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.EdgeConv;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * EdgeConv 测试用例（验证前向传播、维度正确性、反向传播）
 */
public class EdgeConvTest {
    public static void main(String[] args) {
        // 1. 初始化环境
        torch.manual_seed(42);
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

        // 3. 生成测试数据
        // 节点特征 [N, in]
        Tensor x = randn(new long[]{numNodes, inChannels}, floatOpts);
        // 边索引 [2, E]（随机生成无向边）
        long[] edgeData = {
                0,1, 0,2, 1,3, 2,3, 3,4, 4,5, 5,6, 6,7, 7,8, 8,9,
                1,2, 2,4, 4,6, 6,8, 8,0
        };
        Tensor edgeIndex = tensor(edgeData, longOpts).view(2, numEdges);

        // 4. 创建 EdgeConv 实例
        EdgeConv conv = new EdgeConv(inChannels, outChannels);

        // 5. 前向传播测试
        System.out.println("=== 开始前向传播 ===");
        Tensor out = conv.forward(x, edgeIndex);

        // 6. 验证输出维度
        System.out.println("输入维度: " + x.size(0) + " x " + x.size(1));
        System.out.println("输出维度: " + out.size(0) + " x " + out.size(1));
        System.out.println("预期输出维度: " + numNodes + " x " + outChannels);

        // 维度校验
        if (out.size(0) == numNodes && out.size(1) == outChannels) {
            System.out.println("✅ 维度验证通过！");
        } else {
            throw new RuntimeException("❌ 维度验证失败！");
        }

        // 7. 打印输出张量（前5行）
        System.out.println("\n=== 输出张量（前5行） ===");
        printTensor(out.slice(0, new LongOptional(0), new LongOptional(5),1));

        // 8. 反向传播测试（验证梯度链路）
        System.out.println("\n=== 反向传播测试 ===");
        Tensor loss = out.sum();
        loss.backward();
        System.out.println("✅ 反向传播完成，无异常！");

        // 9. 释放资源
        x.close();
        edgeIndex.close();
        out.close();
        loss.close();
        conv.close();
    }

    // 张量打印工具（适配 bytedeco-pytorch）
    private static void printTensor(Tensor tensor) {
        if (tensor == null) {
            System.out.println("Tensor is null");
            return;
        }

        long rows = tensor.size(0);
        long cols = tensor.size(1);
        float[] data = new float[(int) tensor.numel()];
        tensor.detach().data_ptr_float().get(data);

        int idx = 0;
        for (long i = 0; i < rows; i++) {
            System.out.print("  ");
            for (long j = 0; j < cols; j++) {
                System.out.printf("%.4f ", data[idx++]);
                if (j >= 5) { // 只打印前6列，避免过长
                    System.out.print("... ");
                    break;
                }
            }
            System.out.println();
        }
    }
}
