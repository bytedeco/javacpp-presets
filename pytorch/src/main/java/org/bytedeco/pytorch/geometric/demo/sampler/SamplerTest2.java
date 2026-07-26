package org.bytedeco.pytorch.geometric.demo.sampler;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.sampler.BidirectionalNeighborSampler;
import org.bytedeco.pytorch.geometric.sampler.HGTSampler;
import org.bytedeco.pytorch.geometric.sampler.HeteroAdj;
import org.bytedeco.pytorch.geometric.sampler.HeteroSamplerOutput;

import static org.bytedeco.pytorch.global.torch.*;
import java.util.*;

public class SamplerTest2 {

    public static void main(String[] args) {
        System.out.println("=== 启动异构采样器测试 ===");

        try (PointerScope scope = new PointerScope()) {
            // 1. 构造模拟数据 (CSR 格式)
            // 假设 User 0 买了 Item 1; User 1 买了 Item 0 和 1
            HeteroAdj forwardAdj = new HeteroAdj();
            forwardAdj.addEdgeType("user__buys__item",
                    tensor(new long[]{0, 1, 3},new TensorOptions().dtype(new ScalarTypeOptional(kLong()))), // rowPtr
                    tensor(new long[]{1, 0, 1}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())))  // colIndex
            );

            HeteroAdj backwardAdj = new HeteroAdj();
            backwardAdj.addEdgeType("item__bought_by__user",
                    tensor(new long[]{0, 1, 3}, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))), // rowPtr
                    tensor(new long[]{1, 0, 1},new TensorOptions().dtype(new ScalarTypeOptional(kLong())))  // colIndex
            );

            // 2. 测试 BidirectionalNeighborSampler
            testBidirectional(forwardAdj, backwardAdj);

            // 3. 测试 HGTSampler
            testHGT(forwardAdj);
        }
    }

    private static void testBidirectional(HeteroAdj fw, HeteroAdj bw) {
        System.out.println("\n[测试] BidirectionalNeighborSampler...");
        BidirectionalNeighborSampler sampler = new BidirectionalNeighborSampler(fw, bw);

        Map<String, Tensor> seeds = new HashMap<>();
        seeds.put("user", tensor(new long[]{0}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())))); // 以 User 0 为起点

        // 采样参数：正向采 1 个，反向采 1 个
        int[] numNeighbors = {1, 1};
        HeteroSamplerOutput output = sampler.sample(seeds, numNeighbors);

        // 验证结果
        assert output.nodeIds.containsKey("user") : "应当包含 User 类型节点";
        System.out.println("✅ Bidirectional 采样通过！用户节点数: " + output.nodeIds.get("user").size(0));
    }

    private static void testHGT(HeteroAdj adj) {
        System.out.println("\n[测试] HGTSampler (Budget Control)...");
        HGTSampler sampler = new HGTSampler(adj);

        Map<String, Tensor> seeds = new HashMap<>();
        seeds.put("user", tensor(new long[]{0, 1}, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))));

        // 设定 Budget 为 1，即使邻居很多也只能采 1 个
        int nodeBudget = 1;
        int numLayers = 1;
        HeteroSamplerOutput output = sampler.sample(seeds, numLayers, nodeBudget);

        // 验证 Budget 约束
        long sampledCount = output.nodeIds.get("user").size(0);
        if (sampledCount <= nodeBudget + 2) { // +2 是因为包含初始 seed
            System.out.println("✅ HGT Budget 约束生效！采样数: " + sampledCount);
        } else {
            throw new RuntimeException("❌ HGT 采样数超过预算！");
        }
    }
}