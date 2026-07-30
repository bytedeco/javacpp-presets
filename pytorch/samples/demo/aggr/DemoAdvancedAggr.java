package demo.aggr;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.aggr.LSTMAggregation;
import org.bytedeco.pytorch.geometric.aggr.MedianAggregation;
import org.bytedeco.pytorch.geometric.aggr.QuantileAggregation;

import java.util.Arrays;

public class DemoAdvancedAggr {

    public static void main(String[] args) {
        System.out.println("=== Testing Advanced Aggregations (Median, Quantile, LSTM) ===");

        // 构造数据
        // Group 0: [1], [3], [5] -> Median=3
        // Group 1: [2], [4]      -> Median=3
        // Group 2: []            -> Median=0
        long dimSize = 3;

        Tensor x = torch.tensor(new float[]{
                1f, 3f, 5f, // Group 0
                2f, 4f      // Group 1
        }).reshape(5, 1);

        Tensor index = torch.tensor(new long[]{
                0, 0, 0,
                1, 1
        });

        try (PointerScope scope = new PointerScope()) {
            // --- 1. Median ---
            System.out.println("\n[Median org.bytedeco.pytorch.geometric.aggr.Aggregation]");
            MedianAggregation medianAggr = new MedianAggregation();
            Tensor outMedian = medianAggr.forward(x, index, dimSize);
            System.out.println("Values (Expect [3, 3, 0]):\n" + outMedian.flatten());

            // --- 2. Quantile (0.25) ---
            System.out.println("\n[Quantile org.bytedeco.pytorch.geometric.aggr.Aggregation (q=0.25)]");
            // Group 0 (1,3,5): 0.25 quantile should be 1.5 (linear interpolation) or similar
            QuantileAggregation quantAggr = new QuantileAggregation(0.25);
            Tensor outQuant = quantAggr.forward(x, index, dimSize);
            System.out.println("Values:\n" + outQuant.flatten());

            // --- 3. LSTM ---
            System.out.println("\n[LSTM org.bytedeco.pytorch.geometric.aggr.Aggregation]");
            // In=1, Out=4
            LSTMAggregation lstmAggr = new LSTMAggregation(1, 4);
            Tensor outLstm = lstmAggr.forward(x, index, dimSize);
            System.out.println("Output Shape: " + Arrays.toString(outLstm.shape()));
            System.out.println("Values:\n" + outLstm);

            if (outLstm.size(0) == 3 && outLstm.size(1) == 4) {
                System.out.println("PASS: LSTM output shape matches.");
            }
            // 检查第三个节点（无邻居）是否全0
            float zeroCheck = outLstm.select(0, 2).abs().sum().item().toFloat();
            if (zeroCheck == 0) {
                System.out.println("PASS: Isolated node has 0 embedding.");
            } else {
                System.err.println("FAIL: Isolated node should be 0, got " + zeroCheck);
            }
        }
    }
}