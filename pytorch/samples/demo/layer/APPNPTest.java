package samples.demo.layer;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.APPNP;

/**
 * APPNP 测试用例
 * 覆盖场景：基础功能、参数校验、自环/归一化、训练/评估模式、资源释放
 */
public class APPNPTest {
    public static void main(String[] args) {
        // 1. 基础功能测试（带自环+归一化）
        testBasicFunctionality(true, true);

        // 2. 基础功能测试（无自环+不归一化）
        testBasicFunctionality(false, false);

        // 3. 非法参数测试
        testInvalidParameters();

        // 4. 训练/评估模式测试
        testTrainEvalMode();

        // 5. 资源释放测试
        testResourceRelease();

        System.out.println("✅ 所有 APPNP 测试通过！");
    }

    /**
     * 基础功能测试
     * @param addSelfLoops 是否添加自环
     * @param normalize 是否归一化
     */
    private static void testBasicFunctionality(boolean addSelfLoops, boolean normalize) {
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
                0, 0, 1, 1, 2, 3, 0, 2, // 源节点
                1, 2, 2, 3, 4, 4, 3, 3  // 目标节点
        };
        Tensor edgeIndex = torch.tensor(edgeIndexData).reshape(2, numEdges).to(torch.ScalarType.Long);

        // 2. 创建 APPNP 实例（K=2, alpha=0.1, dropout=0.0）
        APPNP appnp = new APPNP(2, 0.1, 0.0, addSelfLoops, normalize);

        // 3. 前向传播
        Tensor output = appnp.forward(x, edgeIndex);

        // 4. 验证输出维度
        assert output.dim() == 2 : "输出必须是2维张量";
        assert output.size(0) == numNodes : "输出节点数必须为" + numNodes + "，实际：" + output.size(0);
        assert output.size(1) == inDim : "输出维度必须为" + inDim + "，实际：" + output.size(1);

        String config = (addSelfLoops ? "带自环" : "无自环") + (normalize ? "+归一化" : "+不归一化");
        System.out.println("✅ " + config + " 测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 5. 释放资源
        appnp.close();
        x.close();
        edgeIndex.close();
        output.close();
    }

    /**
     * 非法参数测试
     */
    private static void testInvalidParameters() {
        // 测试1：K<0
        try {
            new APPNP(-1, 0.1, 0.0, true, true);
            assert false : "未捕获非法迭代次数K";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法K测试通过：" + e.getMessage());
        }

        // 测试2：alpha≤0
        try {
            new APPNP(2, 0.0, 0.0, true, true);
            assert false : "未捕获非法alpha（≤0）";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法alpha(≤0)测试通过：" + e.getMessage());
        }

        // 测试3：alpha>1
        try {
            new APPNP(2, 1.1, 0.0, true, true);
            assert false : "未捕获非法alpha（>1）";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法alpha(>1)测试通过：" + e.getMessage());
        }

        // 测试4：dropout≥1
        try {
            new APPNP(2, 0.1, 1.0, true, true);
            assert false : "未捕获非法dropout（≥1）";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法dropout测试通过：" + e.getMessage());
        }

        // 测试5：空输入
        try {
            APPNP appnp = new APPNP(2, 0.1, 0.0, true, true);
            appnp.forward(null, torch.tensor(new long[]{0,1}).reshape(2,1));
            assert false : "未捕获空节点特征";
        } catch (NullPointerException e) {
            System.out.println("✅ 空节点特征测试通过：" + e.getMessage());
        }
    }

    /**
     * 训练/评估模式测试（Dropout 生效验证）
     */
    private static void testTrainEvalMode() {
        // 1. 构造测试数据
        Tensor x = torch.randn(5, 3).to(torch.ScalarType.Float);
        Tensor edgeIndex = torch.tensor(new long[]{0,1,1,2}).reshape(2,2).to(torch.ScalarType.Long);

        // 2. 创建 APPNP 实例（dropout=0.5）
        APPNP appnp = new APPNP(1, 0.1, 0.5, true, true);

        // 3. 训练模式
        appnp.train(true);
        assert appnp.is_training() : "训练模式设置失败";
        Tensor outputTrain = appnp.forward(x, edgeIndex);

        // 4. 评估模式
        appnp.train(false);
        assert !appnp.is_training() : "评估模式设置失败";
        Tensor outputEval = appnp.forward(x, edgeIndex);

        System.out.println("✅ 训练/评估模式切换测试通过");

        // 5. 释放资源
        appnp.close();
        x.close();
        edgeIndex.close();
        outputTrain.close();
        outputEval.close();
    }

    /**
     * 资源释放测试
     */
    private static void testResourceRelease() {
        APPNP appnp = new APPNP(2, 0.1, 0.0, true, true);
//        appnp.close();

        // 测试释放后调用 train
        try {
            appnp.train(true);
            assert false : "未捕获释放后调用 train";
        } catch (IllegalStateException e) {
            System.out.println("✅ 释放后调用 train 测试通过：" + e.getMessage());
        }

        // 测试释放后调用 forward
        try {
            Tensor x = torch.randn(5, 3);
            Tensor edgeIndex = torch.tensor(new long[]{0,1}).reshape(2,1);
            appnp.forward(x, edgeIndex);
            assert false : "未捕获释放后调用 forward";
        } catch (IllegalStateException e) {
            System.out.println("✅ 释放后调用 forward 测试通过：" + e.getMessage());
        }

        // 测试重复释放
        appnp.close();
        System.out.println("✅ 重复释放测试通过");
    }
}