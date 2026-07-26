package org.bytedeco.pytorch.geometric.demo.transform;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.*;
import java.util.Arrays;
import static org.bytedeco.pytorch.global.torch.*;

public class StructuralTransformTest {
    public static void main(String[] args) {
        System.out.println("=== 启动图结构变换测试 ===");

        // 1. 构造一个 5 节点的极简有向图: 0->1, 1->2, 2->3, 3->4
        Tensor x = zeros(new long[]{5, 2}); // 初始特征 2 维
        long[] edgeArray = {0, 1, 1, 2, 2, 3, 3, 4};
        Tensor edge_index = tensor(edgeArray, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).reshape(2, 4);
        GraphData data = new GraphData(x, edge_index);

        // 2. 组装流水线
        Compose structuralPipeline = new Compose(Arrays.asList(
                new ToUndirected(),   // 转无向: 4条边变8条
                new AddSelfLoops(),   // 加自环: 8条变13条
                new OneHotDegree(5)   // 追加度数: 特征 2 维变 2 + (5+1) = 8 维
        ));

        // 3. 执行
        data = structuralPipeline.apply(data);

        // 4. 验证
        System.out.println("最终边数量: " + data.edge_index.size(1)); // 应为 13
        System.out.println("最终特征维度: " + data.x.size(1));     // 应为 8

        if (data.edge_index.size(1) == 13 && data.x.size(1) == 8) {
            System.out.println("✅ 图结构变换流水线验证成功！");
        }
    }
}