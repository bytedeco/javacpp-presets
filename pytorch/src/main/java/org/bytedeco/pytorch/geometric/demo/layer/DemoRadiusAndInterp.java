package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.pooling.ClusterPooling;

public class DemoRadiusAndInterp {

   public static void main(String[] args) {
        System.out.println("=== Testing Radius Search & KNN Interpolate ===");

        Device device = torch.hasCUDA() ? new Device("cuda") : new Device("cpu");

        try (PointerScope scope = new PointerScope()) {
            // --- 1. Radius Graph ---
            System.out.println("\n[Radius Graph]");
            // 构造一条直线上的点: 0, 1, 2, 5
            Tensor x = torch.tensor(0f, 0f, 0f, // P0
                    1f, 0f, 0f, // P1 (dist 1 to P0)
                    2f, 0f, 0f, // P2 (dist 1 to P1, 2 to P0)
                    5f, 0f, 0f  // P3 (far away)
            ).reshape(4, 3).to(device, torch.ScalarType.Float);
            TensorOptions topt = new TensorOptions().device(new DeviceOptional(device)).dtype(new ScalarTypeOptional(torch.ScalarType.Long));
            Tensor batch = torch.zeros(new long[]{4}, topt);

            // Radius = 1.5
            // Expected edges: (0,1), (1,0), (1,2), (2,1)
            // (0,2) is dist 2 > 1.5, so no edge.
            Tensor edge_index = ClusterPooling.radius_graph(x, 1.5f, batch, false, -1);

            System.out.println("Points: [0, 1, 2, 5]");
            System.out.println("Radius: 1.5");
            System.out.println("Edge Index (Source -> Target):\n" + edge_index);

            // --- 2. KNN Interpolate ---
            System.out.println("\n[KNN Interpolate]");

            TensorOptions tensorOpt = new TensorOptions().device(new DeviceOptional(device)).dtype(new ScalarTypeOptional(torch.ScalarType.Float));
            // Source: 2个点 (0,0) 和 (10,0)
            // Feat:   Blue[0,0,1] 和 Red[1,0,0]
            Tensor srcPos = torch.tensor(0f, 0f, 0f,
                    10f, 0f, 0f).reshape(2, 3).to(device, torch.ScalarType.Float);

            Tensor srcFeat = torch.tensor(0f, 0f, 1f, // Blue
                    1f, 0f, 0f  // Red
            ).reshape(2, 3).to(device, torch.ScalarType.Float);

            // Target: 中点 (5,0)
            Tensor tgtPos = torch.tensor(5f, 0f, 0f).reshape(1, 3).to(device, torch.ScalarType.Float);

            Tensor srcBatch = torch.zeros(new long[]{2}, topt);
            Tensor tgtBatch = torch.zeros(new long[]{1}, topt);

            // k=2, 插值
            Tensor outFeat = ClusterPooling.knn_interpolate(tgtPos, srcPos, srcFeat, tgtBatch, srcBatch, 2);

            System.out.println("Source 1: Pos=0, Color=Blue");
            System.out.println("Source 2: Pos=10, Color=Red");
            System.out.println("Target:   Pos=5");
            System.out.println("Interpolated Color (Expect [0.5, 0, 0.5]):\n" + outFeat);

            // 简单验证
            float r = outFeat.select(1, 0).item().toFloat();
            float b = outFeat.select(1, 2).item().toFloat();

            if (Math.abs(r - 0.5) < 1e-4 && Math.abs(b - 0.5) < 1e-4) {
                System.out.println("PASS: Interpolation logic is correct.");
            } else {
                System.err.println("FAIL: Interpolation incorrect.");
            }
        }
    }
}
