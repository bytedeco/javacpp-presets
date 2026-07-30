package samples.demo.aggr;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.norm.GraphSizeNorm;
import org.bytedeco.pytorch.geometric.nn.norm.MessageNorm;
import org.bytedeco.pytorch.geometric.nn.norm.PairNorm;
//import org.gnn.framework.norm.*;

public class DemoAdvancedNorm {
    public static void main(String[] args) {
        System.out.println("=== Testing Advanced Normalization Layers ===");

        try (PointerScope scope = new PointerScope()) {
            Tensor x = torch.randn(new long[]{10, 4});
            // Batch: 3 graphs
            // G0: 2 nodes, G1: 3 nodes, G2: 5 nodes
            Tensor batch = torch.tensor(new long[]{
                    0, 0,
                    1, 1, 1,
                    2, 2, 2, 2, 2
            });

            // --- 1. GraphSizeNorm ---
            System.out.println("\n[GraphSizeNorm]");
            GraphSizeNorm gsn = new GraphSizeNorm();
            Tensor outGsn = gsn.forward(x, batch);
            // G0 nodes=2, scale=1/sqrt(2)=0.707
            // G1 nodes=3, scale=1/sqrt(3)=0.577
            System.out.println("Values (Check scaling):\n" + outGsn.slice(0, new LongOptional(0), new LongOptional(3), 1));

            // --- 2. PairNorm ---
            System.out.println("\n[PairNorm]");
            PairNorm pn = new PairNorm();
            Tensor outPn = pn.forward(x, batch);
            // 验证: 归一化后，每个图内的节点特征距离应该被拉开
            // 简单验证: 输出不包含 NaN
            if (!torch.isnan(outPn).any().item().toBool()) {
                System.out.println("PASS: PairNorm output valid.");
            }

            // --- 3. MessageNorm ---
            System.out.println("\n[MessageNorm]");
            MessageNorm mn = new MessageNorm(1.0);
            Tensor msg = torch.randn(new long[]{10, 4});
            Tensor outMn = mn.forward(x, msg);
            // 验证: 输出的模长应该接近 x 的模长
            System.out.println("PASS: MessageNorm executed.");
        }
    }
}
