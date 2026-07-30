package samples.demo.layer;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GINConv;
import org.bytedeco.pytorch.geometric.utils.TensorToolkit;

import static org.bytedeco.pytorch.global.torch.*;
import static org.bytedeco.pytorch.geometric.utils.GraphUtils.add_self_loops;

/**
 * GINConv测试用例：验证核心逻辑+数值正确性
 * 测试场景：3节点小图，MLP为单位矩阵（方便手工计算预期值）
 */
public class GINConvTest {

    public static void main(String[] args) {
        // ==============================================
        // 1. 固定测试参数
        // ==============================================
        long N = 3;          // 节点数
        long inChannels = 2; // 输入特征维度
        long outChannels = 2;// 输出特征维度
        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(DeviceType.CPU)));

        // ==============================================
        // 2. 构造测试数据（确定性输入，无随机）
        // ==============================================
        // 边索引：0→1, 1→2（加自环后5条边）
        Tensor edge_index = tensor(new long[]{0, 1, 1, 2}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())))
                .view(2, 2);
        edge_index = add_self_loops(edge_index, N); // 加自环

        // 节点特征：[[1,0], [0,1], [1,1]]
        float[][] xData = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
        float[] flatX = (float[]) TensorToolkit.flatten(xData);
        long[] xShape = TensorToolkit.getShape(xData);
        Tensor x = tensor(flatX, floatOpts).view(xShape);

        // ==============================================
        // 3. 构造MLP（单位矩阵+无偏置，确保MLP(x)=x）
        // ==============================================
        // 构建SequentialImpl：Linear(2,2)（单位矩阵权重）
        LinearImpl linear = new LinearImpl(inChannels, outChannels);
        // 权重设为单位矩阵
        linear.weight().set_data(eye(inChannels, outChannels, floatOpts));
        // 偏置清零
        linear.bias().set_data(zeros(new long[]{outChannels}, floatOpts));
        // 封装为Sequential（GIN的MLP通常是多层，这里简化为单层）
        SequentialImpl mlp = new SequentialImpl();
        mlp.push_back("linear", linear);

        // ==============================================
        // 4. 创建GINConv（eps不可训练，初始值0）
        // ==============================================
        GINConv gin = new GINConv(mlp, false);

        // ==============================================
        // 5. 前向传播
        // ==============================================
        Tensor out = gin.forward(x, edge_index);

        // ==============================================
        // 6. 打印结果（用PyTorch内置print，兼容bytedeco）
        // ==============================================
        System.out.println("===== 输入节点特征 =====");
        torch.print(x);

        System.out.println("\n===== 边索引（加自环后） =====");
        torch.print(edge_index);

        System.out.println("\n===== GINConv输出 =====");
        torch.print(out);

        // ==============================================
        // 7. 手工计算预期值（eps=0时）
        // 核心公式：(1+0)*x + sum(邻居x)
        // 节点0：邻居=自己 → 1*[1,0] + [1,0] = [2,0]
        // 节点1：邻居=0+自己 → 1*[0,1] + [1,0]+[0,1] = [1,2]
        // 节点2：邻居=1+自己 → 1*[1,1] + [0,1]+[1,1] = [2,3]
        // ==============================================
        System.out.println("\n===== 预期输出（eps=0）=====");
        Tensor expected = tensor(new float[]{
                2.0f, 0.0f,
                1.0f, 2.0f,
                2.0f, 3.0f
        }, floatOpts).view(3,2);
        torch.print(expected);

        // ==============================================
        // 8. 验证结果（误差<1e-4）
        // ==============================================
        verifyResult(out, expected);
    }

    /**
     * 验证输出是否符合预期
     */
    private static void verifyResult(Tensor actual, Tensor expected) {
        // 计算误差
        Tensor diff = actual.sub(expected).abs();
        float maxDiff = diff.max().item_float();//Float();

        if (maxDiff < 1e-4) {
            System.out.println("\n✅ GINConv测试通过！输出与预期一致");
        } else {
            System.out.println("\n❌ GINConv测试失败！最大误差：" + maxDiff);
            System.out.println("实际输出：");
            torch.print(actual);
            System.out.println("预期输出：");
            torch.print(expected);
        }
    }
}
