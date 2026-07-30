package samples.demo.aggr;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.aggr.*;
//import org.gnn.framework.aggr.*;

import java.util.Arrays;

public class DemoAggregation {
    public static void main(String[] args) {
        System.out.println("=== Testing PyG-style Aggregations ===");

        // 1. 构造假数据
        // 假设 batch_size=2 (dimSize=2)
        // 5个元素，属于 0 或 1
        // Index: [0, 0, 1, 1, 1]
        long dimSize = 2;
        long features = 4;

        Tensor x = torch.tensor(new float[]{
                1, 1, 1, 1, // Node 0 -> Group 0
                2, 2, 2, 2, // Node 1 -> Group 0
                3, 3, 3, 3, // Node 2 -> Group 1
                4, 4, 4, 4, // Node 3 -> Group 1
                5, 5, 5, 5  // Node 4 -> Group 1
        }).reshape(5, 4);

        Tensor index = torch.tensor(new long[]{0, 0, 1, 1, 1});

        System.out.println("Input X:\n" + x);
        System.out.println("Index: " + index);

        try (PointerScope scope = new PointerScope()) {
            // --- Test 1: Sum ---
            Aggregation sumAggr = new SumAggregation();
            Tensor outSum = sumAggr.forward(x, index, dimSize);
            System.out.println("\n[Sum] Expected: Row0=[3..], Row1=[12..]");
            System.out.println("Actual:\n" + outSum);

            // --- Test 2: Mean ---
            Aggregation meanAggr = new MeanAggregation();
            Tensor outMean = meanAggr.forward(x, index, dimSize);
            System.out.println("\n[Mean] Expected: Row0=[1.5..], Row1=[4..]");
            System.out.println("Actual:\n" + outMean);

            // --- Test 3: Max ---
            Aggregation maxAggr = new MaxAggregation();
            Tensor outMax = maxAggr.forward(x, index, dimSize);
            System.out.println("\n[Max] Expected: Row0=[2..], Row1=[5..]");
            System.out.println("Actual:\n" + outMax);

            // --- Test 4: Softmax org.bytedeco.pytorch.geometric.aggr.Aggregation (Learnable) ---
            System.out.println("\n[Softmax Aggr]");
            SoftmaxAggregation softmaxAggr = new SoftmaxAggregation(features, true);
            Tensor outSoftmax = softmaxAggr.forward(x, index, dimSize);
            System.out.println("Output Shape: " + Arrays.toString(outSoftmax.shape()));
            System.out.println("Values:\n" + outSoftmax);

            // --- Test 5: PowerMean org.bytedeco.pytorch.geometric.aggr.Aggregation (p=1 should be Mean) ---
            System.out.println("\n[PowerMean Aggr (init p=1)]");
            PowerMeanAggregation powerAggr = new PowerMeanAggregation(features, true);
            Tensor outPower = powerAggr.forward(x, index, dimSize);
            System.out.println("Output (Should be close to Mean):\n" + outPower);

            // 验证 PowerMean 是否接近 Mean
            Tensor diff = outPower.sub(outMean).abs().sum();
            if (diff.item().toFloat() < 1e-5) {
                System.out.println("PASS: PowerMean(p=1) equals Mean.");
            } else {
                System.err.println("FAIL: PowerMean(p=1) != Mean. Diff: " + diff.item().toFloat());
            }
        }
    }
}
