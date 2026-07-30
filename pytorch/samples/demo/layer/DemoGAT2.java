package samples.demo.layer;


import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GATConv;
import java.util.Arrays;

public class DemoGAT2 {
    public static void main(String[] args) {
        // 1. 内存管理：使用 PointerScope 确保所有 Tensor 在结束时被回收
        try (PointerScope scope = new PointerScope()) {

            // --- 数据准备 ---
            long numNodes = 10;
            long inChannels = 16;
            long outChannels = 8;
            long heads = 8;

            // 随机生成节点特征 [N, inChannels] -> [10, 16]
            Tensor x = torch.randn(new long[]{numNodes, inChannels});

            // 构建全连接图的 edge_index [2, E]
            // 这里我们模拟一些边，确保有 80 条边左右 (符合你之前报错中的维度)
            Tensor edge_index = createFullEdgeIndex(numNodes);

            System.out.println("Input X Shape: " + Arrays.toString(x.sizes().vec().get()));
            System.out.println("EdgeIndex Shape: " + Arrays.toString(edge_index.sizes().vec().get()));

            // --- 模型初始化 ---
            // 注意：GATConv 内部会自动 register_module 和 register_parameter
            GATConv conv = new GATConv(inChannels, outChannels, heads, 0.2);

            // --- 前向传播 ---
            System.out.println("\n--- Starting Forward Pass ---");

            // 执行前向传播
            // 输出预期维度: [N, outChannels * heads] -> [10, 8 * 8] = [10, 64]
            Tensor output = conv.forward(x, edge_index);

            // --- 结果校验 ---
            System.out.println("Output Shape: " + Arrays.toString(output.sizes().vec().get()));

            if (output.size(0) == numNodes && output.size(1) == (outChannels * heads)) {
                System.out.println("✅ Test Passed: Dimensions are correct!");
            } else {
                System.err.println("❌ Test Failed: Dimension mismatch.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //Input X Shape: [10, 16]
    //EdgeIndex Shape: [2, 90]
    //
    //--- Starting Forward Pass ---
    //Output Shape: [10, 64]
    //✅ Test Passed: Dimensions are correct!
    
    //❌ Test Failed: Dimension mismatch.
    //Output Shape: [10, 8, 8]
    
    /**
     * 辅助方法：创建一个简单的全连接边索引
     */
    private static Tensor createFullEdgeIndex(long numNodes) {
        // 实际上可以用 torch.meshgrid，这里手动构建以保持逻辑清晰
        java.util.List<Long> row = new java.util.ArrayList<>();
        java.util.List<Long> col = new java.util.ArrayList<>();
        for (long i = 0; i < numNodes; i++) {
            for (long j = 0; j < numNodes; j++) {
                if (i != j) { // 排除自环（GATConv 通常在外部或内部处理自环）
                    row.add(i);
                    col.add(j);
                }
            }
        }
        long[] rowArr = row.stream().mapToLong(l -> l).toArray();
        long[] colArr = col.stream().mapToLong(l -> l).toArray();

        Tensor r = torch.tensor(rowArr).unsqueeze(0);
        Tensor c = torch.tensor(colArr).unsqueeze(0);

        // 拼接成 [2, E] 形状的 LongTensor
        return torch.cat(new TensorVector(r, c), 0).to(torch.kLong());
    }
}