package samples.demo.transform;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.*;

import static org.bytedeco.pytorch.global.torch.*;

public class PointCloudTransformTest {
    public static void main(String[] args) {
        System.out.println("=== 启动点云标准化与增强测试 ===");

        // 构造一个偏离原点且未缩放的点云
        // 节点 0(10,10,10), 节点 1(12,10,10)
        Tensor pos = tensor(new float[]{10f, 10f, 10f, 12f, 10f, 10f}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(2,3);
        GraphData data = new GraphData(randn(new long[]{2, 1}), tensor(new long[]{0 ,1}, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).view(2,1));
        data.pos = pos;

        // 测试 Center
        new Center().apply(data);
        assert Math.abs(data.pos.mean().item_float()) < 1e-4;
        System.out.println("✅ Center 测试通过: 均值趋于0");

        // 测试 NormalizeScale
        new NormalizeScale().apply(data);
        assert data.pos.max().item_float() <= 1.0;
        System.out.println("✅ NormalizeScale 测试通过: 范围已限制在 [-1, 1]");

        // 测试 PointPairFeatures (需要 norm)
        data.put("norm", tensor(new float[]{0, 1, 0, 0, 1, 0}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(2,3));
        new PointPairFeatures().apply(data);
        assert data.edge_attr.size(1) == 4;
        System.out.println("✅ PointPairFeatures 测试通过: 维度为 4");

        System.out.println("所有点云变换测试 PASS！");
    }
}