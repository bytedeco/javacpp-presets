package samples.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.ARMAConv;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 最终修复版 ARMAConvTest：统一异常捕获、修正维度测试逻辑
 */
public class ARMAConvTest {
    public static void main(String[] args) {
        try {
            // 1. 基础功能测试（合法参数）
            testBasicFunctionality(2, 3);
            testBasicFunctionality(1, 0);

            // 2. 非法参数测试（核心：统一捕获 IllegalArgumentException）
            testInvalidParameters();

            // 3. 空输入测试
            testNullInput();

            // 4. 资源释放测试
            testResourceRelease();

            System.out.println("✅ 所有 ARMAConv 测试通过！");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 基础功能测试（仅传入合法参数）
     */
    private static void testBasicFunctionality(int numStacks, int numLayers) {
        long inChannels = 4;
        long outChannels = 2;
        long numNodes = 5;
        long numEdges = 8;

        // 构造合法的节点特征（5x4）和边索引（2x8）
        float[] xData = {
                1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f, 16.0f,
                17.0f, 18.0f, 19.0f, 20.0f
        };
        Tensor x = tensor(xData).reshape(numNodes, inChannels).to(ScalarType.Float);

        long[] edgeIndexData = {
                0, 0, 1, 1, 2, 3, 0, 2,
                1, 2, 2, 3, 4, 4, 3, 3
        };
        Tensor edgeIndex = tensor(edgeIndexData).reshape(2, numEdges).to(ScalarType.Long);

        // 创建 ARMAConv 实例（参数合法）
        ARMAConv armaConv = new ARMAConv(inChannels, outChannels, numStacks, numLayers);

        // 测试参数重置
        armaConv.resetParameters();
        System.out.println("✅ 参数重置测试通过（栈数：" + numStacks + "，层数：" + numLayers + "）");

        // 前向传播
        Tensor output = ((ARMAConv)armaConv).forward(x, edgeIndex);

        // 验证输出维度
        assert output.dim() == 2 : "输出必须是2维张量";
        assert output.size(0) == numNodes : "输出节点数必须为" + numNodes + "，实际：" + output.size(0);
        assert output.size(1) == outChannels : "输出维度必须为" + outChannels + "，实际：" + output.size(1);

        String config = "栈数=" + numStacks + ", 层数=" + numLayers;
        System.out.println("✅ " + config + " 测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 释放资源
        armaConv.close();
        x.close();
        edgeIndex.close();
        output.close();
    }

    /**
     * 非法参数测试（核心：统一捕获 IllegalArgumentException，避免触发底层异常）
     */
    private static void testInvalidParameters() {
        // 测试1：输入通道数≤0
        try {
            new ARMAConv(0, 2, 2, 3);
            assert false : "未捕获非法输入通道数";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法输入通道数测试通过：" + e.getMessage());
        }

        // 测试2：输出通道数≤0
        try {
            new ARMAConv(4, -1, 2, 3);
            assert false : "未捕获非法输出通道数";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法输出通道数测试通过：" + e.getMessage());
        }

        // 测试3：栈数量≤0
        try {
            new ARMAConv(4, 2, 0, 3);
            assert false : "未捕获非法栈数量";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法栈数量测试通过：" + e.getMessage());
        }

        // 测试4：层数<0
        try {
            new ARMAConv(4, 2, 2, -1);
            assert false : "未捕获非法层数";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法层数测试通过：" + e.getMessage());
        }

        // 测试5：特征维度不匹配（核心修复：捕获自定义的 IllegalArgumentException）
        ARMAConv conv = null;
        try {
            conv = new ARMAConv(4, 2, 2, 3);
            // 构造维度不匹配的节点特征（1x3）
            Tensor x = tensor(new float[]{1,2,3}).reshape(1,3);
            Tensor edgeIndex = tensor(new long[]{0,1}).reshape(2,1);
            ((ARMAConv)conv).forward(x, edgeIndex);
            assert false : "未捕获维度不匹配";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 特征维度不匹配测试通过：" + e.getMessage());
        } finally {
            // 释放合法创建的实例
            if (conv != null) conv.close();
        }

        // 测试6：边索引维度不匹配
        conv = null;
        try {
            conv = new ARMAConv(4, 2, 2, 3);
            // 构造非法边索引（1x2，而非 2xN）
            Tensor x = tensor(new float[]{1,2,3,4}).reshape(1,4);
            Tensor edgeIndex = tensor(new long[]{0,1}).reshape(1,2);
            ((ARMAConv)conv).forward(x, edgeIndex);
            assert false : "未捕获边索引维度不匹配";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 边索引维度不匹配测试通过：" + e.getMessage());
        } finally {
            if (conv != null) conv.close();
        }
    }

    /**
     * 空输入测试（统一捕获 IllegalArgumentException）
     */
    private static void testNullInput() {
        ARMAConv conv = new ARMAConv(4, 2, 2, 3);

        // 测试1：空节点特征
        try {
            ((ARMAConv)conv).forward(null, tensor(new long[]{0,1}).reshape(2,1));
            assert false : "未捕获空节点特征";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 空节点特征测试通过：" + e.getMessage());
        }

        // 测试2：空边索引
        try {
            Tensor x = tensor(new float[]{1,2,3,4}).reshape(1,4);
            ((ARMAConv)conv).forward(x, (Tensor)null);
            assert false : "未捕获空边索引";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 空边索引测试通过：" + e.getMessage());
        }

        // 释放资源
        conv.close();
    }

    /**
     * 资源释放测试
     */
    private static void testResourceRelease() {
        ARMAConv conv = new ARMAConv(4, 2, 2, 3);
        conv.close();

        // 测试释放后调用 forward
        try {
            Tensor x = tensor(new float[]{1,2,3,4}).reshape(1,4);
            Tensor edgeIndex = tensor(new long[]{0,1}).reshape(2,1);
            ((ARMAConv)conv).forward(x, edgeIndex);
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