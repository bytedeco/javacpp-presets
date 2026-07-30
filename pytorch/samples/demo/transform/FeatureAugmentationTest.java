package samples.demo.transform;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.AdvancedStructureTransforms;

import static org.bytedeco.pytorch.global.torch.*;

public class FeatureAugmentationTest {
    public static void main(String[] args) {
        System.out.println("=== 启动特征补全与图上采样测试 ===");

        // 1. 模拟特征缺失数据：5个节点，其中 2、3 号节点特征为全 0 (缺失)
//        Tensor x = tensor(new float[][]{
//                {1f, 1f}, {2f, 2f}, {0f, 0f}, {0f, 0f}, {5f, 5f}
//        }, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

        Tensor x = tensor(new float[]{
                1f, 1f, 2f, 2f, 0f, 0f, 0f, 0f, 5f, 5f
        }, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(5,2);

        long[] edgeArray = {0, 2, 1, 2, 2, 3, 3, 4}; // 路径图: 0->2, 1->2, 2->3, 3->4
        Tensor edge_index = tensor(edgeArray, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).reshape(2, 4);
        GraphData data = new GraphData(x, edge_index);

        // 2. 执行 FeaturePropagation
        AdvancedStructureTransforms.FeaturePropagation fp = new AdvancedStructureTransforms.FeaturePropagation(40);
        data = fp.apply(data);

        System.out.println("补全后 2 号节点特征 (由 0、1 号扩散而来):");
        System.out.println(data.x.index(new TensorIndexVector(new TensorIndex(tensor(2)))).toString());

        // 3. 测试 HalfHop
        // 节点数应增加 numEdges (4个)，总计 9 个节点
        AdvancedStructureTransforms.HalfHop hh = new AdvancedStructureTransforms.HalfHop();
        data = hh.apply(data);
        System.out.println("Half-Hop 处理后节点总数: " + data.x.size(0));

        if (data.x.index(new TensorIndexVector(new TensorIndex(tensor(2)))).norm().item_float() > 0 && data.x.size(0) == 9) {
            System.out.println("✅ 特征补全与 Half-Hop 验证成功！");
        }
    }
}