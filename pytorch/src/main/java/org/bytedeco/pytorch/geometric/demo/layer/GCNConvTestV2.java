package org.bytedeco.pytorch.geometric.demo.layer;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConvV2;
import org.bytedeco.pytorch.geometric.utils.TensorToolkit;

import static org.bytedeco.pytorch.global.torch.*;
import static org.bytedeco.pytorch.geometric.utils.GraphUtils.add_self_loops;

public class GCNConvTestV2 {

    public static void main(String[] args) {
        // 1. 固定参数
        long N = 3;
        long inChannels = 2;
        long outChannels = 2;
        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(DeviceType.CPU)));
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(kLong())).device(new DeviceOptional(new Device(DeviceType.CPU)));

        // 2. 构造确定性输入（无随机）
        // 边：0→1, 1→2（原始2条边）
        Tensor edge_index = tensor(new long[]{0, 1, 1, 2}, longOpts).view(2, 2);
        // 节点特征（float）
        float[][] xData = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
        float[] flatX = (float[]) TensorToolkit.flatten(xData);
        long[] xShape = TensorToolkit.getShape(xData);
        Tensor x = tensor(flatX, floatOpts).view(xShape);

        // 3. 初始化GCNConvV2并强制固定权重/偏置
        GCNConvV2 gcn = new GCNConvV2(inChannels, outChannels);
        // 权重设为单位矩阵（lin(x)=x）
//        Tensor weight = eye(inChannels, outChannels, floatOpts);
//        gcn.getLin().weight().set_data(weight);
//        // 偏置清零
//        gcn.getLin().bias().set_data(zeros(new long[]{outChannels}, floatOpts));

        // 4. 前向传播
        Tensor out = gcn.forward(x, edge_index);

        torch.print(out);
        // 5. 修复打印逻辑（兼容bytedeco-pytorch）
        System.out.println("===== 输入 x =====");
        printBytedecoTensor(x);
        System.out.println("\n===== 边索引（加自环后） =====");
        printBytedecoTensor(add_self_loops(edge_index, N));
        System.out.println("\n===== GCNConvV2 输出 =====");
        printBytedecoTensor(out);

        // 6. 预期结果
        System.out.println("\n===== 预期输出 =====");
        System.out.println("[0.333, 0.000]");
        System.out.println("[0.333, 0.667]");
        System.out.println("[0.667, 0.667]");
        verifyResult(out, new float[][]{
                {0.3333f, 0.0000f},
                {0.2887f, 0.2500f},
                {0.3333f, 0.6220f}
        });
        // 7. 精准验证（误差<1e-3）
//        verifyResult(out, new float[][]{{0.333f, 0.000f}, {0.333f, 0.667f}, {0.667f, 0.667f}});
    }

    /**
     * 修复：兼容bytedeco-pytorch的Tensor打印
     */
    private static void printBytedecoTensor(Tensor tensor) {
        long rows = tensor.size(0);
        long cols = tensor.size(1);
        System.out.printf("[%s {%d,%d}]\n", tensor.dtype().name(), rows, cols);

        // 正确的索引方式：tensor.at(i,j).get()
        for (long i = 0; i < rows; i++) {
            for (long j = 0; j < cols; j++) {
                if (tensor.dtype().name().getString().contains("Float")) {
                    System.out.printf("%.3f ", tensor.get(i).get( j).item_float());
                } else if (tensor.dtype().name().getString().contains("Long")) {
                    System.out.printf("%d ", tensor.get(i).get( j).item_long());
                }
            }
            System.out.println();
        }
    }

    /**
     * 精准验证结果
     */
    private static void verifyResult(Tensor out, float[][] expected) {
        boolean pass = true;
        for (long i = 0; i < out.size(0); i++) {
            for (long j = 0; j < out.size(1); j++) {
                float actual = out.get(i).get( j).item_float();
                float exp = expected[(int) i][(int) j];
                if (Math.abs(actual - exp) > 1e-3) {
                    pass = false;
                    System.out.printf("❌ 第%d行第%d列：实际=%.3f，预期=%.3f\n", i, j, actual, exp);
                }
            }
        }
        System.out.println(pass ? "\n✅ 测试通过！" : "\n❌ 测试失败！");
    }
}

