package org.bytedeco.pytorch.geometric.demo.transform;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.Delaunay;
import org.bytedeco.pytorch.geometric.transforms.FaceToEdge;
import org.bytedeco.pytorch.geometric.transforms.GridSampling;
import org.bytedeco.pytorch.geometric.transforms.ToSLIC;
//import org.bytedeco.pytorch.geometric.transforms.GeometricTransforms.*;
import static org.bytedeco.pytorch.global.torch.*;

public class TopologyFlattenedTest {
    public static void main(String[] args) {
        System.out.println("=== 启动拓扑变换测试 (严格遵循 Flatten-View 模式) ===");

        // --- 1. 测试 FaceToEdge ---
        // 原始数据：1个面，顶点为 0, 1, 2
        float[] posMeshFlat = {0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f};
        long[] faceFlat = {0L, 1L, 2L};

        Tensor posMesh = tensor(posMeshFlat, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(3, 3);
        Tensor face = tensor(faceFlat, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).view(3, 1); // [3, num_faces]

        GraphData meshData = new GraphData(randn(new long[]{3, 4}), null);
        meshData.pos = posMesh;
        meshData.put("face", face);

        new FaceToEdge(true).apply(meshData);
        // 一个三角形面 -> 3条无向边 = 6个 entry
        assert meshData.edge_index.size(1) == 6;
        System.out.println("✅ FaceToEdge Pass: 边数=" + meshData.edge_index.size(1));


        // --- 2. 测试 Delaunay (Gabriel Graph 实现) ---
        // 原始数据：4个点形成正方形
        float[] posPointsFlat = {0f, 0f, 1f, 0f, 1f, 1f, 0f, 1f};
        Tensor posPoints = tensor(posPointsFlat, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(4, 2);

        GraphData pointData = new GraphData(randn(new long[]{4, 4}), null);
        pointData.pos = posPoints;

        new Delaunay().apply(pointData);
        // 正方形应该至少有 4 条边 (8 个 entry)
        assert pointData.edge_index.size(1) >= 8;
        System.out.println("✅ Delaunay Pass: 生成边数=" + pointData.edge_index.size(1));


        // --- 3. 测试 GridSampling ---
        // 原始数据：4个点，前两个在同一体素，后两个在另一体素
        float[] posClustersFlat = {0.01f, 0.01f, 0.02f, 0.02f, 0.50f, 0.50f, 0.51f, 0.51f};
        Tensor posClusters = tensor(posClustersFlat, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(4, 2);

        GraphData clusterData = new GraphData(randn(new long[]{4, 4}), null);
        clusterData.pos = posClusters;

        new GridSampling(0.1f).apply(clusterData);
        // 4个点应被采样为 2个
        assert clusterData.numNodes() == 2;
        System.out.println("✅ GridSampling Pass: 剩余节点数=" + clusterData.numNodes());


        // --- 4. 测试 ToSLIC ---
        // 模拟 4x4 的像素点阵 (16个点)
        float[] imgPosFlat = new float[32];
        for(int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                imgPosFlat[(i*4+j)*2] = (float)i;
                imgPosFlat[(i*4+j)*2+1] = (float)j;
            }
        }
        Tensor imgPos = tensor(imgPosFlat, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(16, 2);
        Tensor imgCol = zeros(new long[]{16, 3}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))); // 纯黑图像

        GraphData imgData = new GraphData(imgCol, null);
        imgData.pos = imgPos;

        // 聚类为 4 个超像素
        new ToSLIC(4, 1.0f).apply(imgData);
        assert imgData.numNodes() == 4;
        assert imgData.edge_index != null;
        System.out.println("✅ ToSLIC Pass: 节点数=" + imgData.numNodes());

        System.out.println("\n🎉 所有严格 Flatten 模式测试全部通过！");
    }
}
