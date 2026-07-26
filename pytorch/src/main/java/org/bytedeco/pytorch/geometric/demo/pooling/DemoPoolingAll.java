package org.bytedeco.pytorch.geometric.demo.pooling;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.pooling.ClusterPooling;
import org.bytedeco.pytorch.geometric.nn.pooling.EdgePooling;
import org.bytedeco.pytorch.geometric.nn.pooling.EdgePoolingOutput;
import org.bytedeco.pytorch.geometric.nn.pooling.SAGPooling;

import java.util.Arrays;

import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.zeros;

public class DemoPoolingAll {
    public static void main(String[] args) {
        System.out.println("=== Testing Advanced Pooling Layers ===");

        Device device = new Device("cpu"); // 聚类算法涉及 CPU 数组操作，暂用 CPU 演示

        try (PointerScope scope = new PointerScope()) {
            Tensor x = randn(10, 4).to(device, torch.ScalarType.Float);
            Tensor edge_index = torch.tensor(new long[]{
                    0, 1, 1, 2, 2, 3, 3, 4, // Line graph 0-1-2-3-4
                    5, 6, 6, 7, 7, 8, 8, 9  // Line graph 5-6-7-8-9
            }).reshape(2, 8).to(device, torch.ScalarType.Long);
            Tensor batch = torch.tensor(new long[]{0,0,0,0,0, 1,1,1,1,1}).to(device, torch.ScalarType.Long);
            // 2. Edge Pooling
            System.out.println("\n[EdgePooling]");
            EdgePooling ep = new EdgePooling(4);
            ep.to(device, true);
            EdgePoolingOutput epRet = ep.edgePool(x, edge_index);
            System.out.println("EdgePooling Nodes: " + epRet.x.size(0));


            // 1. Cluster Pooling (Graclus)
            System.out.println("\n[Graclus + MaxPool]");
            Tensor cluster = ClusterPooling.graclus(edge_index, 10);
            System.out.println("Clusters: " + cluster);
            Tensor xPool = ClusterPooling.max_pool(cluster, x);
            System.out.println("Pooled X Shape: " + Arrays.toString(xPool.shape()));


            // 3. SAG Pooling
            System.out.println("\n[SAG Pooling]");
            SAGPooling sag = new SAGPooling(4, 0.5);
            sag.to(device,true);
            // 这里假设 SAGPooling 继承并重写了 forward
            // Tensor[] sagRet = sag.forward(x, edge_index, batch);
            // System.out.println("SAG Pooled Nodes: " + sagRet[0].size(0));

            testEdgePooling();
        }
    }


    public static void testEdgePooling() {
        System.out.println("\r\n=== Starting EdgePooling Test ===");

        // 1. 初始化参数
        long inChannels = 8;
        long numNodes = 6;

        // 构造一个简单的图
        // 0 -- 1 (边分数高)
        // 2 -- 3
        // 4 -- 5
        // 0 -- 2 (跨连边)

        // 2. 构造输入特征 [6, 8]
        Tensor x = randn(new long[]{numNodes, inChannels},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float))).set_requires_grad(true);

        // 3. 构造边索引 [2, 8] (无向图，每条边写两次)
        long[] edgeData = {
                0, 1, 1, 0, // 0-1
                2, 3, 3, 2, // 2-3
                4, 5, 5, 4, // 4-5
                0, 2, 2, 0  // 0-2
        };
        Tensor edge_index = torch.tensor(edgeData,
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long))).view(2, 8);

        // 4. 构造 Batch [6] (假设所有节点属于同一个图)
        Tensor batch = zeros(new long[]{numNodes},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        try {
            // 5. 初始化模型
            EdgePooling pool = new EdgePooling(inChannels);

            // 6. 前向传播
            System.out.println("Running forward pass...");
            EdgePoolingOutput output = pool.edgePool(x, edge_index);

            // 7. 结果验证
            Tensor xNew = output.x;
            Tensor cluster = output.cluster;

            System.out.println("\n--- Test Results ---");
            System.out.println("Original Node Count: " + numNodes);
            System.out.println("Pooled Node Count: " + xNew.size(0));
            System.out.println("New Feature Shape: " + Arrays.toString(xNew.shape()));

            // Cluster 应该是一个长度为 6 的向量，值代表旧节点所属的新节点 ID
            System.out.println("Cluster Mapping: " + cluster);

            // 逻辑校验：池化后的节点数应该小于原始节点数
            if (xNew.size(0) < numNodes && xNew.size(0) >= numNodes / 2) {
                System.out.println("\nSUCCESS: Graph successfully coarsened.");
            } else {
                System.out.println("\nWARNING: Node count reduction unexpected.");
            }

            // 8. 检查梯度流 (测试反向传播是否崩溃)
            System.out.println("\nTesting backward pass...");
            Tensor loss = xNew.sum();
            loss.backward();
            System.out.println("SUCCESS: Backward pass completed without SIGSEGV.");

        } catch (Exception e) {
            System.err.println("Test Failed!");
            e.printStackTrace();
        }
    }
}