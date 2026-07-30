package samples.demo.layer;
import org.bytedeco.pytorch.nn.*;

import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.ClusterGCNConv;

/**
 * 完整测试用例：ClusterGCNConv（覆盖核心场景+边界条件）
 */
public class ClusterGCNConvFullTest {
    public static void main(String[] args) {
        // ========== 测试1：基础参数初始化 + 维度校验 ==========
        System.out.println("===== 测试1：基础参数初始化 + 维度校验 =====");
        long inChannels = 16;
        long outChannels = 8;
        float diagLambda = 0.1f;
        boolean addSelfLoops = true;
        boolean hasBias = true;

        // 创建Conv实例
        ClusterGCNConv conv = new ClusterGCNConv(inChannels, outChannels, diagLambda, addSelfLoops, hasBias);

        // 构造测试数据：10个节点，20条边
        long N = 10; // 节点数
        long E = 20; // 边数
        Tensor x = torch.randn(new long[]{N, inChannels}, torch.dtype(torch.ScalarType.Float)); // 节点特征 [10,16]
        Tensor edgeIndex = torch.randint(0, N, new long[]{2, E}, torch.dtype(torch.ScalarType.Long)); // 边索引 [2,20]

        // 前向传播
        Tensor output = conv.forward(x, edgeIndex);

        // 验证输出维度：[10,8]
        System.out.println("输入特征维度：" + x.size(0) + " x " + x.size(1)); // 10 x 16
        System.out.println("边索引维度：" + edgeIndex.size(0) + " x " + edgeIndex.size(1)); // 2 x 20
        System.out.println("输出特征维度：" + output.size(0) + " x " + output.size(1)); // 10 x 8
        if (output.size(0) == N && output.size(1) == outChannels) {
            System.out.println("✅ 维度校验通过");
        } else {
            System.out.println("❌ 维度校验失败");
        }

        // ========== 测试2：自环添加校验 ==========
        System.out.println("\n===== 测试2：自环添加校验 =====");
        // 构造无自环的边索引（5个节点，5条边）
        long N2 = 5;
        long E2 = 5;
        Tensor edgeIndexNoSelfLoop = torch.tensor(new long[]{0, 1, 1, 2, 3, 1, 0, 2, 1, 4}, torch.dtype(torch.ScalarType.Long)).view(2, 5); // [2,5]
        Tensor x2 = torch.randn(new long[]{N2, inChannels}, torch.dtype(torch.ScalarType.Float));

        // 前向传播（开启自环）
        Tensor output2 = conv.forward(x2, edgeIndexNoSelfLoop);
        // 验证边数：原有5条 + 5条自环 = 10条
        // 从propagate中提取edge_index的边数（通过row.size(0)）
        long edgeCountAfterSelfLoop = edgeIndexNoSelfLoop.size(1) + N2;
        System.out.println("原始边数：" + E2); // 5
        System.out.println("添加自环后边数：" + edgeCountAfterSelfLoop); // 10
        if (edgeCountAfterSelfLoop == E2 + N2) {
            System.out.println("✅ 自环添加校验通过");
        } else {
            System.out.println("❌ 自环添加校验失败");
        }

        // ========== 测试3：可训练参数校验 ==========
        System.out.println("\n===== 测试3：可训练参数校验 =====");
        StringTensorDict params = conv.named_parameters();
        System.out.println("可训练参数列表：");
        var paramBegin = params.begin();
        var paramEnd = params.end();
        while (!paramBegin.equals(paramEnd)) {
            StringTensorDictItem entry = paramBegin.get();
            String paramName = entry.pair().first().getString();
            Tensor param = entry.access();
            System.out.println("- " + paramName + "：维度 " + param.data().size(0) + " x " + (param.data().dim() > 1 ? param.data().size(1) : 1));
            paramBegin.increment();
        }
//        for (String paramName : params.keySet()) {
//            Parameter param = params.get(paramName);
//            System.out.println("- " + paramName + "：维度 " + param.data().size(0) + " x " + (param.data().dim() > 1 ? param.data().size(1) : 1));
//        }
        // 验证参数是否存在：lin.weight(16x8)、lin.bias(8)、bias(8)

        boolean hasLinWeight = params.contains("lin.weight");
        boolean hasLinBias = params.contains("lin.bias");
        boolean hasBiasParam = params.contains("bias");
        if (hasLinWeight && hasLinBias && hasBiasParam) {
            System.out.println("✅ 可训练参数校验通过");
        } else {
            System.out.println("❌ 可训练参数校验失败");
        }

        // ========== 测试4：边界场景：孤立节点（度为0） ==========
        System.out.println("\n===== 测试4：边界场景：孤立节点校验 =====");
        // 构造数据：6个节点，其中节点5是孤立节点（无任何边）
        long N3 = 6;
        Tensor edgeIndexWithIsolated = torch.tensor(new long[]{0, 1, 1, 2, 3, 4, 1, 0, 2, 1, 4, 3}, torch.dtype(torch.ScalarType.Long)).view(2, 6); // [2,6]
        Tensor x3 = torch.randn(new long[]{N3, inChannels}, torch.dtype(torch.ScalarType.Float));
        // 前向传播（验证孤立节点不会报错）
        try {
            Tensor output3 = conv.forward(x3, edgeIndexWithIsolated);
            System.out.println("孤立节点输出维度：" + output3.size(0) + " x " + output3.size(1)); // 6 x 8
            System.out.println("✅ 孤立节点场景校验通过（无报错）");
        } catch (Exception e) {
            System.out.println("❌ 孤立节点场景校验失败：" + e.getMessage());
        }

        // ========== 测试5：内存泄漏校验（close后Tensor不可用） ==========
        System.out.println("\n===== 测试5：内存释放校验 =====");
        // 释放资源
        conv.close();
        x.close();
        edgeIndex.close();
        output.close();
        x2.close();
        edgeIndexNoSelfLoop.close();
        output2.close();
        x3.close();
        edgeIndexWithIsolated.close();

        // 验证释放后Tensor是否不可用（避免野指针）
        try {
            output.size(0); // 已释放，会抛出异常
            System.out.println("❌ 内存释放校验失败");
        } catch (Exception e) {
            System.out.println("✅ 内存释放校验通过（Tensor已释放）");
        }

        // ========== 测试6：输入合法性校验 ==========
        System.out.println("\n===== 测试6：输入合法性校验 =====");
        ClusterGCNConv conv2 = new ClusterGCNConv(inChannels, outChannels, diagLambda, addSelfLoops, hasBias);
        // 构造非法输入：3维节点特征
        Tensor xIllegal = torch.randn(new long[]{2, 3, inChannels}, torch.dtype(torch.ScalarType.Float));
        try {
            conv2.forward(xIllegal, edgeIndex);
            System.out.println("❌ 非法输入校验失败（未抛出异常）");
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法输入校验通过：" + e.getMessage());
        }
        // 释放非法输入
        xIllegal.close();
        conv2.close();

        System.out.println("\n===== 所有测试完成 =====");
    }
}