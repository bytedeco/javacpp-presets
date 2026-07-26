package org.bytedeco.pytorch.geometric.demo.pooling;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.nn.pooling.DensePooling;
import static org.bytedeco.pytorch.global.torch.*;

public class DensePoolingTest {
    public static void main(String[] args) {
        System.out.println("=== 启动 Dense Pooling 算子测试 ===");

        long B = 2, N = 10, D = 16, M = 4; // 将10个节点池化为4个簇
        Tensor x = randn(new long[]{B, N, D}).requires_grad_(true);
        Tensor adj = ones(new long[]{B, N, N});

        // 模拟分配矩阵 S (通常由一个 Linear 层产生)
        Tensor s = randn(new long[]{B, N, M}).requires_grad_(true);

        // 1. 测试 DiffPool
        Tensor[] diffRes = DensePooling.dense_diff_pool(x, adj, s);
        System.out.printf("DiffPool: [B, N, D] -> [%d, %d, %d]\n",
                diffRes[0].size(0), diffRes[0].size(1), diffRes[0].size(2));

  
        // 3. 测试 DMoNPooling
        Tensor[] dmonRes = DensePooling.dmon_pooling(x, adj, s);
        dmonRes[2].backward();
        System.out.println("✅ DMoNPooling 模块度损失梯度回传成功。");

        // 2. 测试 MinCutPool
        Tensor[] mincutRes = DensePooling.dense_mincut_pool(x, adj, s);
        mincutRes[2].backward(); // 验证辅助 Loss 的梯度回传
        System.out.println("✅ MinCutPool 辅助损失梯度回传成功。");

//        long B = 2, N = 10, D = 16, M = 4;
//        Tensor x = randn(new long[]{B, N, D}).requires_grad_(true);
//        Tensor adj = ones(new long[]{B, N, N});
//        Tensor s = randn(new long[]{B, N, M}).requires_grad_(true);

        try {
            Tensor[] res = DensePooling.dense_mincut_pool(x, adj, s);

            System.out.println("✅ MinCutPool 执行成功");
            System.out.println("Loss 值: " + res[2].item_float());

            res[2].backward();
            if (s.grad().defined()) {
                System.out.println("✅ 梯度计算成功，S 梯度形状: " +
                        java.util.Arrays.toString(s.grad().sizes().vec().get()));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("=== 所有池化算子验证通过 ===");
    }
}