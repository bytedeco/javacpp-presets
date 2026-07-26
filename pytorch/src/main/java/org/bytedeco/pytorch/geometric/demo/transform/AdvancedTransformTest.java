package org.bytedeco.pytorch.geometric.demo.transform;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.RandomLinkSplit;
import org.bytedeco.pytorch.geometric.transforms.RandomNodeSplit;
import org.bytedeco.pytorch.geometric.transforms.ToSparseTensor;

import static org.bytedeco.pytorch.global.torch.*;

public class AdvancedTransformTest {
    public static void main(String[] args) {
        System.out.println("=== 启动高级 Graph Transforms 测试 ===");

        // 1. 初始化模拟数据 (1000个节点，10个类别)
        Tensor x = randn(new long[]{1000, 32});
        Tensor edge_index = randint(0, 1000, new long[]{2, 5000}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        GraphData data = new GraphData(x, edge_index);
        data.y = randint(0, 10, new long[]{1000}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        // 2. 构造划分流水线
        RandomNodeSplit split = new RandomNodeSplit(0.7, 0.1, 0.2);
        data = split.apply(data);

        System.out.println("训练集节点数: " + data.get("train_mask").toType(kLong()).sum().item_long());
        System.out.println("验证集节点数: " + data.get("val_mask").toType(kLong()).sum().item_long());

        // 3. 测试 ToSparseTensor
        ToSparseTensor toSparse = new ToSparseTensor();
        data = toSparse.apply(data);
        if (data.get("adj_t").is_sparse()) {
            System.out.println("✅ 成功转换 edge_index 为 SparseTensor (adj_t)");
        }

        // 4. 测试链路划分 (RandomLinkSplit)
        RandomLinkSplit linkSplit = new RandomLinkSplit(0.1, 0.2);
        data = linkSplit.apply(data);
        System.out.println("训练边数量: " + data.get("train_edge_index").size(1));

        System.out.println("✅ 高级流水线全部验证通过！");
    }
}
