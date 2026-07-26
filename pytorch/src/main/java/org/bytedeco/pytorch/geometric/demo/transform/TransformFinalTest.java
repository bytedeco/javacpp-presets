package org.bytedeco.pytorch.geometric.demo.transform;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.*;
import java.util.Arrays;
import static org.bytedeco.pytorch.global.torch.*;

public class TransformFinalTest {
    public static void main(String[] args) {
        System.out.println("=== 启动可解释性与过滤 Transforms 测试 ===");

        // 1. 准备数据：1000个节点，类别为 0-4
        Tensor x = randn(new long[]{1000, 16});
        Tensor y = randint(0, 5, new long[]{1000}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        Tensor edge_index = randint(0, 1000, new long[]{2, 2000}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        GraphData data = new GraphData(x, edge_index);
        data.y = y;

        // 初始化训练掩码，全部设为 true
        Tensor train_mask = ones(new long[]{1000}, new TensorOptions().dtype(new ScalarTypeOptional(kBool())));

        // 2. 测试 RemoveTrainingClasses
        // 我们想移除类别 3 和 4 的训练数据
        System.out.println("移除前训练集节点总数: " + train_mask.toType(kLong()).sum().item_long());

        RemoveTrainingClasses remover = new RemoveTrainingClasses(Arrays.asList(3, 4));
        data = remover.apply(data);

        System.out.println("移除后训练集节点总数: " + train_mask.toType(kLong()).sum().item_long());

        // 3. 验证类别 3 的节点是否真的被移除了
        Tensor checkCls3 = logical_and(train_mask, data.y.eq(new Scalar(3)));
        if (checkCls3.toType(kLong()).sum().item_long() == 0) {
            System.out.println("✅ 成功移除：类别 3 的节点不再出现在训练集中。");
        }

        // 4. 测试 ComposeFilters
        // 定义一个过滤器：只保留边数大于 100 的子图
        GraphFilter edgeFilter = d -> d.edge_index.size(1) > 100;
        // 定义一个过滤器：只保留具有训练掩码的图
        GraphFilter maskFilter = d -> train_mask != null;

        ComposeFilters filterCompose = new ComposeFilters(Arrays.asList(edgeFilter, maskFilter));

        if (filterCompose.apply(data)) {
            System.out.println("✅ ComposeFilters 验证通过：该数据符合所有过滤规则。");
        }
    }
}
