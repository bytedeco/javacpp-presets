package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConvV2;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConvV3;
import org.bytedeco.pytorch.geometric.utils.TensorToolkit;

import static org.bytedeco.pytorch.global.torch.*;
import static org.bytedeco.pytorch.geometric.utils.GraphUtils.add_self_loops;

public class GCNConvTest {

    public static void main(String[] args) {
        // ==============================================
        // 1. 构造极小图：3个节点，2条边（全float类型）
        // ==============================================
        long N = 3; // 节点数
        long inChannels = 2;  // 输入特征维度
        long outChannels = 2; // 输出维度

        // 边：0→1, 1→2（Long类型，使用new TensorOptions()创建）
        Tensor edge_index = torch.tensor(new long[]{
                0, 1,
                1, 2
        }, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).view(2, 2);

        // 固定输入特征（float类型）
        float[][] dd = {
                {1.0f, 0.0f},
                {0.0f, 1.0f},
                {1.0f, 1.0f}
        };
        float[] flatdd = (float[]) TensorToolkit.flatten(dd);
        long[] shape = TensorToolkit.getShape(dd);
        Tensor x = torch.tensor(flatdd, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(shape);

        // ==============================================
        // 2. 手动加自环（GCN必须）
        // ==============================================
        edge_index = add_self_loops(edge_index, N);

        // ==============================================
        // 3. 创建GCNConvV2，强制权重为单位矩阵+偏置清零
        // ==============================================
        GCNConvV3 gcn = new GCNConvV3(inChannels, outChannels);

        // 把线性层权重设为单位矩阵（float类型），确保lin(x)=x
        Tensor weight = eye(inChannels, outChannels, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        gcn.getLin().weight().set_data(weight);
        // 偏置清零（关键：避免偏置影响结果）
        var opt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        gcn.getLin().bias().set_data(zeros(new long[]{outChannels},opt ));

        // ==============================================
        // 4. 前向传播（确保输入为float类型）
        // ==============================================
        Tensor out = gcn.forward(x, edge_index);

        // ==============================================
        // 5. 格式化打印输入 & 输出，对比结果
        // ==============================================
        System.out.println("===== 输入 x（float类型） =====");
        printTensor(x);

        System.out.println("\n===== 边索引（加自环后） =====");
        printTensor(edge_index);

        System.out.println("\n===== GCNConvV2 输出（float类型） =====");
        printTensor(out);
        torch.print(out);

        // ==============================================
        // 6. 预期结果（手工计算的float近似值）
        // ==============================================
        System.out.println("\n===== 预期输出（近似）=====");
        System.out.println("[0.333, 0.000]");
        System.out.println("[0.333, 0.666]");
        System.out.println("[0.666, 0.666]");

        // ==============================================
        // 7. 自动验证结果（误差<1e-3）
        // ==============================================
        verifyResult(out, new float[][]{{0.333f, 0.000f}, {0.333f, 0.666f}, {0.666f, 0.666f}});
    }

    // 辅助方法：格式化打印Tensor（适配float类型）
    private static void printTensor(Tensor tensor) {
        long rows = tensor.size(0);
        long cols = tensor.size(1);
        System.out.printf("[%s {%d,%d}]\n", tensor.dtype().name(), rows, cols);
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

    // 辅助方法：自动验证结果是否符合预期（float误差<1e-3）
    private static void verifyResult(Tensor out, float[][] expected) {
        boolean pass = true;
        long n = out.size(0);
        long f = out.size(1);

        if (n != expected.length || f != expected[0].length) {
            pass = false;
        } else {
            for (long i = 0; i < n; i++) {
                for (long j = 0; j < f; j++) {
                    float actual = out.get(i).get( j).item_float();
                    float exp = expected[(int) i][(int) j];
                    if (Math.abs(actual - exp) > 1e-3) {
                        pass = false;
                        System.out.printf("\n❌ 第%d行第%d列不符：实际=%.3f，预期=%.3f\n",
                                i, j, actual, exp);
                    }
                }
            }
        }

        if (pass) {
            System.out.println("\n✅ 测试通过！GCNConvV2输出和预期一致");
        } else {
            System.out.println("\n❌ 测试失败！输出和预期不符");
        }
    }
}