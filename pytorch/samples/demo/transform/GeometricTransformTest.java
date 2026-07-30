package samples.demo.transform;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.*;
//import org.bytedeco.pytorch.geometric.transforms.GeometricTransforms.*;
import static org.bytedeco.pytorch.global.torch.*;

public class GeometricTransformTest {
    public static void main(String[] args) {
        System.out.println("=== 启动几何变换 (Vision Transforms) 联合测试 ===");

        // 1. 构造基础数据
        // 节点 0 在原点 (0,0,0)，节点 1 在 (1,1,1)
        Tensor pos = tensor(new float[]{0, 0, 0, 1, 1, 1}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(2,3);
        // 边：0 -> 1
        Tensor edge_index = tensor(new long[]{0, 1}, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).view(2,1);

        GraphData data = new GraphData(randn(new long[]{2, 16}), edge_index);
        data.pos = pos;

        // --- 测试 1: Distance ---
        new Distance(false).apply(data);
        float distVal = data.edge_attr.item_float();
        // sqrt(1^2 + 1^2 + 1^2) = sqrt(3) ≈ 1.732
        assert Math.abs(distVal - Math.sqrt(3)) < 1e-4;
        System.out.println("✅ Distance 测试通过: " + distVal);
        data.edge_attr = null; // 重置

        // --- 测试 2: Cartesian ---
        new Cartesian().apply(data);
        // 预期的相对位移是 (1, 1, 1) 经过归一化处理
        // 注意：Cartesian 实现中通常会做偏移处理以适应图像坐标
        System.out.println("✅ Cartesian 结果维度: " + java.util.Arrays.toString(data.edge_attr.sizes().vec().get()));
        assert data.edge_attr.size(1) == 3;
        data.edge_attr = null;

        // --- 测试 3: LocalCartesian ---
        new LocalCartesian().apply(data);
        assert data.edge_attr.size(1) == 3;
        System.out.println("✅ LocalCartesian 测试通过");
        data.edge_attr = null;

        // --- 测试 4: Polar (仅限 2D) ---
        // 构造 2D 数据：点 0(0,0), 点 1(1,0) -> 距离 1, 角度 0
        GraphData data2d = new GraphData(randn(new long[]{2, 4}), edge_index);
        data2d.pos = tensor(new float[]{0, 0, 1, 0}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(2,2);
        new Polar().apply(data2d);
        // Polar 返回 [r, theta]，theta 归一化到 [0, 1]，0度对应 0.5
        float r = data2d.edge_attr.index(new TensorIndexVector(new TensorIndex(0), new TensorIndex(0))).item_float();
        float theta = data2d.edge_attr.index(new TensorIndexVector(new TensorIndex(0), new TensorIndex(1))).item_float();
        assert Math.abs(r - 1.0) < 1e-4;
        assert Math.abs(theta - 0.5) < 1e-4;
        System.out.println("✅ Polar 测试通过: r=" + r + ", theta=" + theta);

        // --- 测试 5: Spherical (仅限 3D) ---
        // 构造 3D 数据：点 0(0,0,0), 点 1(0,0,1) -> z轴正方向
        GraphData data3d = new GraphData(randn(new long[]{2, 4}), edge_index);
        data3d.pos = tensor(new float[]{0, 0, 0, 0, 0, 1}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(2,3);
        new Spherical().apply(data3d);
        // Spherical 返回 [r, phi, theta]
        // 对于 (0,0,1): r=1, phi=0.5 (x,y均为0时atan2为0), theta=0 (acos(1)/pi)
        float sr = data3d.edge_attr.index(new TensorIndexVector(new TensorIndex(0), new TensorIndex(0))).item_float();
        float s_theta = data3d.edge_attr.index(new TensorIndexVector(new TensorIndex(0), new TensorIndex(2))).item_float();
        assert Math.abs(sr - 1.0) < 1e-4;
        assert Math.abs(s_theta - 0.0) < 1e-4;
        System.out.println("✅ Spherical 测试通过: r=" + sr + ", theta_elev=" + s_theta);

        

        System.out.println("\n恭喜！所有几何变换测试全部 PASS！🚀");
    }
}