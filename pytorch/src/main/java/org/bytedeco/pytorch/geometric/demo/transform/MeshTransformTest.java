package org.bytedeco.pytorch.geometric.demo.transform;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.GenerateMeshNormals;
import org.bytedeco.pytorch.geometric.transforms.SamplePoints;

import static org.bytedeco.pytorch.global.torch.*;

public class MeshTransformTest {
    public static void main(String[] args) {
        System.out.println("=== 启动网格与采样变换测试 ===");

        // 构造一个简单的金字塔网格 (5个点, 4个三角形面)
        Tensor pos = tensor(new float[]{0,0,0,1,0,0, 0,1,0, 1,1,0, 0.5f, 0.5f, 1}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(5,3);
        Tensor face = tensor(new long[]{0,1,4, 1,3,4, 3,2,4, 2,0,4}, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).view(4,3).t();

        GraphData data = new GraphData(randn(new long[]{5, 8}), null);
        data.pos = pos;
        data.put("face", face);

        // 1. 测试面转边
//        new FaceToEdge(true).call(data);
//        assert data.edge_index != null && data.edge_index.size(1) > 0;
//        System.out.println("✅ FaceToEdge 转换成功, 边数: " + data.edge_index.size(1));

        // 2. 测试法向量生成
        new GenerateMeshNormals().apply(data);
        assert data.get("norm").size(0) == 5;
        System.out.println("✅ GenerateMeshNormals 生成成功");

        // 3. 测试网格采样
        new SamplePoints(100).apply(data);
        assert data.pos.size(0) == 100;
        System.out.println("✅ SamplePoints 采样成功, 得到 100 个点");
    }
}