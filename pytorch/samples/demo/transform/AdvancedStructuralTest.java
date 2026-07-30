package samples.demo.transform;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.AdvancedStructuralTransforms;

import static org.bytedeco.pytorch.global.torch.*;

public class AdvancedStructuralTest {
    public static void main(String[] args) {
        System.out.println("=== 启动 SIGN 与 GCNNorm 联合测试 ===");

        // 1. 构造 4 节点图
        Tensor x = randn(new long[]{4, 8}); // 8维原始特征
        long[] edgeArray = {0, 1, 1, 2, 2, 3, 3, 0};
        Tensor edge_index = tensor(edgeArray, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).reshape(2, 4);
        GraphData data = new GraphData(x, edge_index);

        // 2. 执行 SIGN 变换 (预计算 2 阶)
        // 结果特征维度应为 8 * (2 + 1) = 24 维
        AdvancedStructuralTransforms.SIGN signTransform = new AdvancedStructuralTransforms.SIGN(2);
        data = signTransform.apply(data);

        System.out.println("SIGN 处理后特征维度: " + data.x.size(1));

        // 3. 测试 ToDense
        data = new AdvancedStructuralTransforms.ToDense().apply(data);
        System.out.println("稠密邻接矩阵形状: " + java.util.Arrays.toString(data.adj.sizes().vec().get()));

        if (data.x.size(1) == 24 && data.adj.size(0) == 4) {
            System.out.println("✅ SIGN 与 ToDense 预处理流水线验证成功！");
        }
    }
}