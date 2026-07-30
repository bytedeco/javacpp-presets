package samples.demo.pooling;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.model.KNNIndex;
import org.bytedeco.pytorch.geometric.nn.model.L2KNNIndex;
import org.bytedeco.pytorch.geometric.nn.pooling.GlobalPooling;
import org.bytedeco.pytorch.geometric.nn.pooling.SAGPooling;
import org.bytedeco.pytorch.geometric.nn.pooling.TopKPooling;

import java.util.Arrays;

public class DemoPooling {

    public static void main(String[] args) {
        System.out.println("=== Testing Pooling Layers ===");

//        Device device = torch_cuda.is_available() ? new Device("cuda") : new Device("cpu");
        if (!torch.hasCUDA()) {
            System.out.println("===(CUDA) not support ===");
//            throw new RuntimeException("Need CUDA for this enterprise demo!");
        }
        Device device = new Device("mps");// new Device("cuda");
        try (PointerScope scope = new PointerScope()) {
            // Data Setup
            long N = 10;
            long D = 4;
            Tensor x = torch.randn(new long[]{N, D}).to(device, torch.ScalarType.Float);
            Tensor batch = torch.tensor(new long[]{0,0,0,0,1,1,1,1,1,1}).to(device, torch.ScalarType.Long); // 4 nodes in G0, 6 in G1

            // --- 1. Global Pooling ---
            System.out.println("\n[Global Pooling]");
            Tensor outAdd = GlobalPooling.global_add_pool(x, batch);
            System.out.println("Add Pool Shape: " + Arrays.toString(outAdd.shape())); // Expect [2, 4]

            // --- 2. KNN Index ---
            System.out.println("\n[KNN Index (L2)]");
            KNNIndex knn = new L2KNNIndex(3); // k=3
            Tensor[] searchRet = knn.search(x, null, batch, batch); // Search within same batch
            Tensor dist = searchRet[0];
            Tensor idx = searchRet[1];
            System.out.println("KNN Indices Shape: " + Arrays.toString(idx.shape())); // [10, 3]
            // Check self-loop (distance should be 0)
            System.out.println("First node dists (Expect first is 0): " + dist.select(0, 0));

            // --- 3. TopK Pooling ---
            System.out.println("\n[TopK Pooling]");
            // Construct simple edges: 0-1, 1-2, 2-3, 4-5
            Tensor edge_index = torch.tensor(new long[]{
                    0, 1, 2, 4,
                    1, 2, 3, 5
            }).reshape(2, 4).to(device, torch.ScalarType.Long);

            TopKPooling topk = new TopKPooling(D, 0.5); // Keep 50% = 5 nodes
            topk.to(device,true);

            Tensor[] poolRet = topk.topk(x, edge_index, batch);
            Tensor xNew = poolRet[0];
            Tensor edgeNew = poolRet[1];
            Tensor batchNew = poolRet[2];
            Tensor perm = poolRet[3];

            System.out.println("Original Nodes: " + N);
            System.out.println("Pooled Nodes: " + xNew.size(0)); // Should be 5
            System.out.println("Pooled Batch: " + batchNew);
            System.out.println("New edge_index:\n" + edgeNew);
            // 验证边是否重连：如果节点 0 和 1 都被保留，且原来有边，则新 edge_index 应包含 (newId0, newId1)

            testSAGPooling();
        }
    }


    public static void testSAGPooling() {
        System.out.println("=== Testing SAGPooling ===");
        long N = 10;
        long C = 16;

        Tensor x = torch.randn(N, C);
        Tensor edge_index = torch.tensor(new long[]{
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                1, 2, 3, 4, 5, 6, 7, 8, 9
        }).reshape(2, 9);

        // 初始化池化层，保留 50% 节点
        SAGPooling sag = new SAGPooling(C, 0.5);

        // 前向传播
        Tensor[] result = ((SAGPooling)sag).sagPool(x, edge_index, (Tensor)null);

        System.out.println("Pooled X shape: " + Arrays.toString(result[0].shape())); // 应该是 [5, 16]
        System.out.println("New edge_index shape: " + Arrays.toString(result[1].shape()));
        System.out.println("Survival Indices: " + result[3]);
    }
}
