package samples.demo.transform;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.*;
import static org.bytedeco.pytorch.global.torch.*;

public class TopologyTransformTest {
    public static void main(String[] args) {
        System.out.println("=== 启动拓扑与空间图变换测试 ===");

        // 1. 模拟 10 个节点在 2D 空间的位置
        Tensor pos = rand(new long[]{10, 2});
        Tensor x = randn(new long[]{10, 16});
        GraphData data = new GraphData(x, null);
        data.pos = pos;

        // 2. 构造流水线：KNN构建 -> 添加自环 -> 移除自环
        Compose pipeline = new Compose(java.util.Arrays.asList(
                new TopologyTransforms.KNNGraph(3),        // 每个点找最近 3 个邻居
                new TopologyTransforms.AddRemainingSelfLoops() // 补全自环
        ));

        data = pipeline.apply(data);
        System.out.println("KNN 构建后的边数 (含自环): " + data.edge_index.size(1));

        // 3. 测试 RemoveSelfLoops
        data = new TopologyTransforms.RemoveSelfLoops().apply(data);
        System.out.println("移除自环后的边数: " + data.edge_index.size(1));

        // 4. 验证维度
        if (data.edge_index.size(0) == 2) {
            System.out.println("✅ 拓扑结构变换验证成功！");
        }
    }
}
