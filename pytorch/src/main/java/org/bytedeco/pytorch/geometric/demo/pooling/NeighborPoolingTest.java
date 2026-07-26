package org.bytedeco.pytorch.geometric.demo.pooling;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.nn.pooling.GraphNeighborPooling;

import java.util.Arrays;

import static org.bytedeco.pytorch.global.torch.*;

public class NeighborPoolingTest {
    public static void main(String[] args) {
        System.out.println("=== 启动邻域池化算子测试 ===");

        // 1. 构造简单图
        // 节点 0 特征 10, 节点 1 特征 20, 节点 2 特征 30
        Tensor x = tensor(new float[]{10f, 20f, 30f}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).reshape(3, 1);

        x.requires_grad_();
        // 边：0 -> 1, 1 -> 2 (无向图需考虑双向，这里模拟有向聚合)
        long[][] edges = {{0, 1}, {1, 2}};
        long[] flattened = Arrays.stream(edges)
                .flatMapToLong(Arrays::stream)
                .toArray();
        Tensor edge_index = tensor(flattened, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).reshape(2,2);

        // 2. 测试 Max Pooling
        // 对于节点 1，其邻居有 0，加上自身 1，最大值应为 max(10, 20) = 20
        // 对于节点 2，其邻居有 1，加上自身 2，最大值应为 max(20, 30) = 30
        Tensor xMax = GraphNeighborPooling.max_pool_neighbor_x(x, edge_index);
        System.out.println("Max Pool 结果:\n" + Arrays.toString(xMax.sizes().vec().get()));

        // 3. 测试 Avg Pooling
        // 对于节点 1，均值应为 (10 + 20) / 2 = 15
        Tensor xAvg = GraphNeighborPooling.avg_pool_neighbor_x(x, edge_index);
        System.out.println("Avg Pool 结果:\n" + Arrays.toString(xAvg.sizes().vec().get()));

        // 4. 验证反向传播
        xMax.sum().backward();
        if (x.grad().defined()) {
            System.out.println("✅ 邻域聚合梯度回传成功。");
        }
    }
}