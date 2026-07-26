package org.bytedeco.pytorch.geometric.demo.transform;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.aggr.GRUAggregation;
import org.bytedeco.pytorch.geometric.aggr.PatchTransformerAggregation;

import java.util.Arrays;

public class DemoGruAndTransformer {


    public static void main(String[] args) {
        System.out.println("=== Testing GRU & PatchTransformer Aggregations ===");

        // 数据构造:
        // Node 0: Neighbors [1, 1], [2, 2] (Seq len 2)
        // Node 1: Neighbors [3, 3]         (Seq len 1)
        // Node 2: No neighbors             (Seq len 0)
        long dimSize = 3;
        long channels = 2;

        Tensor x = torch.tensor(new float[]{
                1, 1,
                2, 2,
                3, 3
        }).reshape(3, 2);

        Tensor index = torch.tensor(new long[]{
                0, 0, 1
        });

        try (PointerScope scope = new PointerScope()) {
            // --- 1. GRU org.bytedeco.pytorch.geometric.aggr.Aggregation ---
            System.out.println("\n[GRU org.bytedeco.pytorch.geometric.aggr.Aggregation]");
            // In=2, Out=4
            GRUAggregation gruAggr = new GRUAggregation(2, 4);
            Tensor outGru = gruAggr.forward(x, index, dimSize);
            System.out.println("Output Shape: " + Arrays.toString(outGru.shape()));

            // 检查 Node 2 (无邻居) 是否全 0
            float zeroCheck = outGru.select(0, 2).abs().sum().item().toFloat();
            if (zeroCheck == 0) {
                System.out.println("PASS: Isolated node output is 0.");
            } else {
                System.err.println("FAIL: Isolated node output is " + zeroCheck);
            }

            // --- 2. PatchTransformer org.bytedeco.pytorch.geometric.aggr.Aggregation ---
            System.out.println("\n[PatchTransformer org.bytedeco.pytorch.geometric.aggr.Aggregation]");
            // 2 channels, 2 heads, 1 layer
            PatchTransformerAggregation ptAggr = new PatchTransformerAggregation(channels, 2, 1);

            // Transformer 初始化有随机性，我们只检查形状和运行是否 Crash
            Tensor outPt = ptAggr.forward(x, index, dimSize);
            System.out.println("Output Shape: " + Arrays.toString(outPt.shape()));
            System.out.println("Values:\n" + outPt);

            // 验证维度是否保持 (Sum org.bytedeco.pytorch.geometric.aggr.Aggregation 不改变特征维度)
            if (outPt.size(1) == channels) {
                System.out.println("PASS: Feature dimension preserved.");
            }

            // 再次检查 Node 2
            float zeroCheckPt = outPt.select(0, 2).abs().sum().item().toFloat();
            if (zeroCheckPt == 0) {
                System.out.println("PASS: Isolated node output is 0.");
            } else {
                System.err.println("FAIL: Isolated node output is " + zeroCheckPt);
            }
        }
    }
}