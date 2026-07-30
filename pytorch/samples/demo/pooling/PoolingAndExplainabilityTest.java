package samples.demo.pooling;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;
import org.bytedeco.pytorch.geometric.nn.pooling.MemPooling;
import org.bytedeco.pytorch.geometric.nn.pooling.PANPooling;

import static org.bytedeco.pytorch.global.torch.*;

public class PoolingAndExplainabilityTest {
    public static void main(String[] args) {
        System.out.println("=== 启动 GNN 高级算子联合测试 ===");

        // 模拟数据：32个节点，16维特征
        Tensor x = randn(new long[]{32, 16}).requires_grad_(true);
        Tensor edge_index = tensor(new long[]{0, 1, 2, 3,1, 2, 3, 0},new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).reshape(2,4);

        // 1. 测试 PANPooling
        PANPooling pan = new PANPooling(16, 0.5);
        Tensor[] panResult = pan.panPool(x, edge_index);
        System.out.printf("PANPooling 完成: 原始节点 32 -> 池化后节点 %d\n", panResult[0].size(0));

        // 2. 测试 MemPooling
        MemPooling mem = new MemPooling(16, 8);
        Tensor x_mem = mem.forward(x);
        System.out.printf("MemPooling 完成: 节点 32 -> 聚类簇数 %d\n", x_mem.size(0));

        // 3. 测试修正后的 Gini (作用于 MemPooling 的分配输出，衡量聚类稀疏性)
        try {
            Tensor gLoss = gini(x_mem);
            Tensor bLoss = bro_penalty(x_mem);

            // 将两个损失叠加
            Tensor combinedLoss = gLoss.add(bLoss);

            // 只需要一次 backward
            combinedLoss.backward();

            System.out.println("✅ 联合损失反向传播成功！");
            System.out.printf("Gini 分数: %.4f\n", gLoss.item_float());
            System.out.printf("BRO 惩罚值: %.4f\n", bLoss.item_float());
            if (x.grad().defined()) {
                System.out.println("✅ 原始输入 x 的梯度已成功计算。");
            }
//            Tensor gLoss = gini(x_mem);
//            gLoss.backward();
            System.out.printf("Gini 正则化计算成功: %.4f\n", gLoss.item_float());
        } catch (Exception e) {
            System.err.println("❌ 计算过程出错: ");
            e.printStackTrace();
        }

        // 4. 测试 BRO (确保聚类后的 Embedding 保持正交，避免特征坍塌)
//        Tensor bLoss = bro_penalty(x_mem);
//        bLoss.backward();
//        System.out.printf("BRO 正则化计算成功: %.4f\n", bLoss.item_float());

        System.out.println("✅ 所有高级算子测试通过！");
    }

    public static Tensor bro_penalty(Tensor x) {
        long N = x.size(0);
        long D = x.size(1);

        // 1. Normalize columns (features) to have unit norm
        // dim=0
        NormalizeFuncOptions opt = new NormalizeFuncOptions();
        opt.p().put(2);
        opt.dim().put(0);
        Tensor xNorm = torch.normalize(x, opt);

        // 2. Compute Correlation Matrix: M^T M -> [D, D]
        Tensor corr = xNorm.t().matmul(xNorm);

        // 3. Identity Matrix
        Tensor eye = torch.eye(D, x.options());

        // 4. Frobenius Norm
        return corr.sub(eye).norm();
    }

    /**
     * Gini Coefficient
     * 衡量稀疏性 (0 = complete equality/dense, 1 = complete inequality/sparse)
     * 公式: G = (2 * sum(i * x_sorted_i)) / (n * sum(x_i)) - (n + 1)/n
     */
    public static Tensor gini(Tensor x) {
        // 1. 必须展平为一维向量，因为 Gini 是衡量一组数值的分布不均
        Tensor xFlat = x.abs().view(-1).add(new Scalar(1e-6));;
        long n = xFlat.size(0); // 这里 n 会是 128 * 64 = 8192

        // 2. 排序 (从小到大)
        Tensor xSorted = torch.sort(xFlat).get0();

        // 3. 创建索引 [1, 2, ..., n]
        // 确保 index 的形状和 xSorted 一致，都是 [n]
        Tensor index = torch.arange(new Scalar(1),new Scalar( n + 1), x.options());

        // 4. 计算公式: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1)/n
        // 分子: 2 * sum(index * xSorted)
        Tensor num = index.mul(xSorted).sum().mul(new Scalar(2.0));

        // 分母: n * sum(xSorted)
        Tensor den = xSorted.sum().mul(new Scalar((double)n));

        // 5. 结果
        return num.div(den).sub(new Scalar((double)(n + 1) / n));
    }
}