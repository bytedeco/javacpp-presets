package samples.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.AGNNConv;

/**
 * AGNNConv 测试用例
 */
public class AGNNConvTest {
    
    public static void main(String[] args) {
        // 1. 基础功能测试（可学习 beta）
        testBasicFunctionality(true);

        // 2. 基础功能测试（不可学习 beta）
        testBasicFunctionality(false);

        // 3. 非法参数测试
        testInvalidParameters();

        // 4. 资源释放测试
        testResourceRelease();

        System.out.println("✅ 所有 AGNNConv 测试通过！");
    }

    /**
     * 基础功能测试
     * @param requiresGrad 是否启用 beta 梯度
     */
    private static void testBasicFunctionality(boolean requiresGrad) {
        // 1. 构造测试数据
        long numNodes = 5;    // 5个节点
        long inDim = 3;       // 输入维度3
        long numEdges = 8;    // 8条边

        // 节点特征 [5, 3]
        float[] xData = {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f
        };
        Tensor x = torch.tensor(xData).reshape(numNodes, inDim).to(torch.ScalarType.Float);

        // 边索引 [2, 8]
        long[] edgeIndexData = {
                0, 0, 1, 1, 2, 3, 0, 2, // 源节点（j）
                1, 2, 2, 3, 4, 4, 3, 3  // 目标节点（i）
        };
        Tensor edgeIndex = torch.tensor(edgeIndexData).reshape(2, numEdges).to(torch.ScalarType.Long);

        // 2. 创建 AGNNConv 实例
        AGNNConv agnnConv = new AGNNConv(requiresGrad);

        // 3. 前向传播
        Tensor output = ((AGNNConv)agnnConv).forward(x, edgeIndex);

        // 4. 验证输出维度
        assert output.dim() == 2 : "输出必须是2维张量";
        assert output.size(0) == numNodes : "输出节点数必须为" + numNodes + "，实际：" + output.size(0);
        assert output.size(1) == inDim : "输出维度必须为" + inDim + "，实际：" + output.size(1);

        String gradStatus = requiresGrad ? "可学习" : "不可学习";
        System.out.println("✅ " + gradStatus + " beta 测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 5. 测试参数重置
        agnnConv.resetParameters();
        System.out.println("✅ " + gradStatus + " beta 参数重置测试通过");

        // 6. 释放资源
        agnnConv.close();
        x.close();
        edgeIndex.close();
        output.close();
    }

    /**
     * 非法参数测试
     */
    private static void testInvalidParameters() {
        AGNNConv conv = new AGNNConv(true);

        // 测试1：空节点特征
        try {
            ((AGNNConv)conv).forward(null, torch.tensor(new long[]{0,1}).reshape(2,1));
            assert false : "未捕获空节点特征";
        } catch (NullPointerException e) {
            System.out.println("✅ 空节点特征测试通过：" + e.getMessage());
        }

        // 测试2：空边索引
        try {
            Tensor x = torch.randn(3, 2);
            ((AGNNConv)conv).forward(x, (Tensor)null);
            assert false : "未捕获空边索引";
        } catch (NullPointerException e) {
            System.out.println("✅ 空边索引测试通过：" + e.getMessage());
        }

        // 测试3：1维节点特征
        try {
            Tensor x = torch.randn(5); // 1维张量（非法）
            Tensor edgeIndex = torch.tensor(new long[]{0,1}).reshape(2,1);
            ((AGNNConv)conv).forward(x, edgeIndex);
            assert false : "未捕获1维节点特征";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 1维节点特征测试通过：" + e.getMessage());
        }

        // 测试4：边索引形状错误（3行）
        try {
            Tensor x = torch.randn(5, 3);
            Tensor edgeIndex = torch.tensor(new long[]{0,1,2}).reshape(3,1); // 非法
            ((AGNNConv)conv).forward(x, edgeIndex);
            assert false : "未捕获边索引形状错误";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 边索引形状错误测试通过：" + e.getMessage());
        }

        // 释放资源
        conv.close();
    }

    /**
     * 资源释放测试
     */
    private static void testResourceRelease() {
        AGNNConv conv = new AGNNConv(true);
        conv.close();

        // 测试释放后调用 forward
        try {
            Tensor x = torch.randn(5, 3);
            Tensor edgeIndex = torch.tensor(new long[]{0,1}).reshape(2,1);
            ((AGNNConv)conv).forward(x, edgeIndex);
            assert false : "未捕获释放后调用 forward";
        } catch (IllegalStateException e) {
            System.out.println("✅ 释放后调用 forward 测试通过：" + e.getMessage());
        }

        // 测试释放后重置参数
        try {
            conv.resetParameters();
            assert false : "未捕获释放后重置参数";
        } catch (IllegalStateException e) {
            System.out.println("✅ 释放后重置参数测试通过：" + e.getMessage());
        }

        // 测试重复释放
        conv.close();
        System.out.println("✅ 重复释放测试通过");
    }
}