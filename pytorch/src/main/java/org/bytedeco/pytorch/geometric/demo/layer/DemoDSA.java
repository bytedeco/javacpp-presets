package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.aggr.*;

import java.util.Arrays;

public class DemoDSA {

   public static void main(String[] args) {
        System.out.println("=== Testing DSA (Degree Scaler org.bytedeco.pytorch.geometric.aggr.Aggregation) Components ===");

        long dimSize = 2; // Batch=2
        long features = 4;

        // Data: 
        // Graph 0: 3 nodes
        // Graph 1: 2 nodes
        Tensor x = torch.randn(5, features);
        Tensor index = torch.tensor(new long[]{0, 0, 0, 1, 1});

        try (PointerScope scope = new PointerScope()) {
            // 1. 构建基础聚合器组合: Mean, Max, Min, Std
            MultiAggregation aggregators = new MultiAggregation(
                    new MeanAggregation(),
                    new MaxAggregation(),
                    new MinAggregation(),
                    new StdAggregation()
            );

//            System.out.println("Aggregators: " + aggregators);
            // 2. 构建 Degree Scaler
            // 假设 avg_deg = 2.5
            // Scalers: Identity, Amplification, Attenuation
            DegreeScalerAggregation dsa = new DegreeScalerAggregation(
                    2.5,
                    Arrays.asList("identity", "amplification", "attenuation"),
                    aggregators
            );

            System.out.println("DSA Structure initialized.");

            // 3. Forward
            Tensor out = dsa.forward(x, index, dimSize);

            // 4. Check Output Shape
            // 4 aggregators * 3 scalers * features = 12 * features
            System.out.println("Input Shape: " + Arrays.toString(x.shape()));
            System.out.println("Output Shape: " + Arrays.toString(out.shape()));

            long expectedFeatures = 4 * 3 * features;
            if (out.size(1) == expectedFeatures) {
                System.out.println("PASS: Output feature dimension matches PNA formula.");
            } else {
                System.err.println("FAIL: Expected " + expectedFeatures + " features. but Got " + out.size(1) + ".");
            }

            System.out.println("Output Sample:\n" + out.slice(1, new LongOptional(0), new LongOptional(4), 1));
        }
    }
}
