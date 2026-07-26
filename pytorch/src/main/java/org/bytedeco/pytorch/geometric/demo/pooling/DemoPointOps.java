package org.bytedeco.pytorch.geometric.demo.pooling;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.pooling.ClusterPooling;
import org.bytedeco.pytorch.geometric.nn.pooling.VoxelOutput;

import java.util.Arrays;

public class DemoPointOps {

    public static void main(String[] args) {
        System.out.println("=== Testing Point Cloud Operations ===");

        Device device = torch.hasCUDA() ? new Device("cuda") : new Device("cpu");
        System.out.println("Device: " + (device.is_cuda() ? "CUDA" : "CPU"));

        try (PointerScope scope = new PointerScope()) {
            // 构造数据: 2个 Sample，每个 100 点，3D
            long B = 2;
            long N = 100;
            long D = 3;

            // 对于 FPS，输入通常是 [B, N, D]
            Tensor xBatched = torch.randn(new long[]{B, N, D}).to(device,torch.ScalarType.Float);

            // 对于 KNN/Voxel，通常输入是 Flattened [B*N, D] + Batch Index
            Tensor xFlat = xBatched.reshape(B * N, D);
            TensorOptions tensorOpt = new TensorOptions().device(new DeviceOptional(device)).dtype( new ScalarTypeOptional(torch.ScalarType.Float));
            Tensor batch = torch.arange(new Scalar(B), tensorOpt).unsqueeze(1).repeat(new long[]{1, N}).reshape(B * N);

            // --- 1. FPS ---
            System.out.println("\n[FPS]");
            int numSamples = 10;
            Tensor fpsIdx = ClusterPooling.fps(xBatched, numSamples);
            System.out.println("FPS Output Shape: " + Arrays.toString(fpsIdx.shape())); // [2, 10]
            System.out.println("Indices:\n" + fpsIdx);

            // --- 2. KNN Graph ---
            System.out.println("\n[KNN Graph (k=5)]");
            Tensor edge_index = ClusterPooling.knn_graph(xFlat, 5, batch);
            System.out.println("Edge Index Shape: " + Arrays.toString(edge_index.shape())); // [2, 2*100*5] = [2, 1000]

            // --- 3. Approx KNN ---
            System.out.println("\n[Approx KNN Graph (proj=1)]");
            // 投影到 1D 做 KNN (极端情况测试)
            Tensor approxedge_index = ClusterPooling.approx_knn_graph(xFlat, 5, batch, 1);
            System.out.println("Approx Edge Index Shape: " + Arrays.toString(approxedge_index.shape()));

            // --- 4. Voxel Grid ---
            System.out.println("\n[Voxel Grid Downsampling]");
            // 范围大概在 -3 ~ 3，设 voxel_size = 1.0，应该能聚合不少点
            double voxelSize = 1.0d;
            VoxelOutput voxelRet = ClusterPooling.voxel_grid(xFlat, batch, voxelSize);
            Tensor xPool = voxelRet.pos;
            Tensor batchPool = voxelRet.batch;

            System.out.println("Original Points: " + (B * N));
            System.out.println(" pooled Points: " + xPool.size(0));
            System.out.println("Pooled X Shape: " + Arrays.toString(xPool.shape()));
            if (batchPool != null) {
                System.out.println("Pooled Batch Shape: " + Arrays.toString(batchPool.shape()));
            }
        }
    }
}
