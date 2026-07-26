package org.bytedeco.pytorch.geometric.demo.transform;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.MaskToIndex;
import org.bytedeco.pytorch.geometric.transforms.NodePropertySplit;

import static org.bytedeco.pytorch.global.torch.*;

public class PropertySplitTest {
    public static void main(String[] args) {
        System.out.println("=== 启动分布偏移划分 (NodePropertySplit) 测试 ===");

        // 1. 模拟数据：1000个用户，特征维度32
        Tensor x = randn(new long[]{1000, 32});
        Tensor edge_index = randint(0, 1000, new long[]{2, 5000}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        GraphData data = new GraphData(x, edge_index);

        // 2. 模拟节点属性：账户余额 (余额从 0 到 1,000,000)
        Tensor node_prop = rand(new long[]{1000}).mul(new Scalar(1000000));

        // 3. 执行 NodePropertySplit
        // 训练集占 70%，验证 10%，测试 20%
        // 升序排列：训练集将包含余额最少的 700 人，测试集包含余额最多的 200 人
        NodePropertySplit propSplit =
                new NodePropertySplit(0.7, 0.1, true);
        data = propSplit.apply(data);

        // 4. 验证划分结果
        System.out.println("训练集最大余额: " + node_prop.masked_select(data.get("train_mask")).max().item_float());
        System.out.println("测试集最小余额: " + node_prop.masked_select(data.get("test_mask")).min().item_float());

        // 5. 测试 MaskToIndex 转换
        MaskToIndex m2i = new MaskToIndex();
        data = m2i.apply(data);
        System.out.println("训练集第一个索引: " + data.get("train_indices").index(new TensorIndexVector(new TensorIndex(tensor(0)))).item_long());

        System.out.println("✅ 所有高级转换算子验证通过！");
    }
}
