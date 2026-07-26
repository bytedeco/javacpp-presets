package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.TransformerConvV2;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * TransformerConv 测试用例（验证前向传播和维度正确性）
 */
public class TransformerConvTest {
    public static void main(String[] args) {
        // 1. 初始化环境
        torch.manual_seed(42);
        Device cpu = new Device(kCPU());
        TensorOptions opts = new TensorOptions()
                .dtype(new ScalarTypeOptional(kFloat()))
                .device(new DeviceOptional(cpu))
                .requires_grad(new BoolOptional(true));

        // 2. 测试参数
        long inChannels = 8;    // 输入通道
        long outChannels = 4;   // 每个头的输出通道
        long heads = 2;         // 注意力头数
        long numNodes = 5;      // 节点数
        long numEdges = 8;      // 边数

        // 3. 生成测试数据
        // 节点特征 [N, in]
        Tensor x = randn(new long[]{numNodes, inChannels}, opts);
        // 边索引 [2, E]
        long[] edgeData = {0,1,0,2,1,3,2,3,3,4,4,0,1,2,2,4};
        Tensor edgeIndex = tensor(edgeData, new TensorOptions().dtype(new ScalarTypeOptional(kLong())).device(new DeviceOptional(cpu))).view(2, numEdges);

        // 4. 创建 TransformerConv 实例
        TransformerConvV2 conv = new TransformerConvV2(inChannels, outChannels, heads);

        // 5. 前向传播
        System.out.println("=== 开始前向传播 ===");
        Tensor out = conv.forward(x, edgeIndex);

        // 6. 验证输出维度
        System.out.println("输入维度: " + x.size(0) + " x " + x.size(1));
        System.out.println("输出维度: " + out.size(0) + " x " + out.size(1));
        System.out.println("预期输出维度: " + numNodes + " x " + (heads * outChannels));

        // 7. 验证维度正确性
        if (out.size(0) == numNodes && out.size(1) == heads * outChannels) {
            System.out.println("✅ 维度验证通过！");
        } else {
            throw new RuntimeException("❌ 维度验证失败！");
        }

        // 8. 打印输出张量（前3行）
        System.out.println("\n=== 输出张量（前3行） ===");
        printTensor(out.slice(0,new LongOptional( 0), new LongOptional(3),1));

        // 9. 反向传播测试（验证梯度）
        System.out.println("\n=== 反向传播测试 ===");
        Tensor loss = out.sum();
        loss.backward();
        System.out.println("✅ 反向传播完成，无异常！");

        // 10. 释放资源
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
                if (j >= 5) { // 只打印前6列
                    System.out.print("... ");
                    break;
                }
            }
            System.out.println();
        }
    }
}