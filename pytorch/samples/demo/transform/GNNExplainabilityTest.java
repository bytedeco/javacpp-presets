package samples.demo.transform;
import org.bytedeco.pytorch.nn.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Functional;

import static org.bytedeco.pytorch.global.torch.*;

public class GNNExplainabilityTest {

    public static void main(String[] args) {
        // 1. 模拟 GNN 最后一层输出的节点表示 (Batch Size: 128, Hidden Dim: 64)
        // 必须开启 requires_grad 以测试反向传播
        Tensor representations = randn(new long[]{128, 64}).requires_grad_(true);

        System.out.println("=== 开始测试可解释性正则化算子 ===");

        // 2. 测试 BRO (Batch Representation Orthogonality)
        // 原理：Loss = ||H * H^T - I||_F^2
     
        
        try {
            Tensor broLoss = Functional.bro_penalty(representations); // calculateBRO(representations);
            System.out.printf("BRO Penalty 输出: %.4f\n", broLoss.item_float());

            // 验证梯度
            broLoss.backward();
            if (representations.grad().defined()) {
                System.out.println("✅ BRO 梯度反向传播成功。");
            }
        } catch (Exception e) {
            System.err.println("❌ BRO 计算失败: " + e.getMessage());
        }

        // 3. 测试 Gini 系数 (Induced Sparsity)
        // 原理：用于衡量特征分布的稀疏性，Gini 越高，表示只有少数特征起关键作用
        try {
            // 清空梯度
            representations.grad().zero_();

//            Tensor giniLoss = calculateGini(representations);
            Tensor giniLoss = Functional.gini(representations);
            System.out.printf("Gini Coefficient 输出: %.4f\n", giniLoss.item_float());

            // 验证梯度
            giniLoss.backward();
            if (representations.grad().defined()) {
                System.out.println("✅ Gini 梯度反向传播成功。");
            }
        } catch (Exception e) {
            System.err.println("❌ Gini 计算失败: " + e.getMessage());
        }

        System.out.println("=== 测试完成 ===");
    }

    /**
     * BRO 实现参考：让 Batch 内的表示尽可能正交
     */
    public static Tensor calculateBRO(Tensor h) {
        long n = h.size(0);
        // 计算 Gram 矩阵 H * H^T
        Tensor gram = mm(h, h.transpose(0, 1));
        // 生成单位矩阵
        Tensor identity = eye(n, h.options());
        // 计算 Frobenius 范数的平方
        return norm(sub(gram, identity), new Scalar(2));
    }

    /**
     * Gini 实现参考：诱导特征选择的稀疏性
     */
    public static Tensor calculateGini(Tensor h) {
        // 1. 取绝对值并排序
        Tensor v = h.abs().view(-1).sort().get0(); //.get(0);
        long n = v.size(0);

        // 2. 计算 Gini 公式索引
        Tensor index = linspace(new Scalar(1), new Scalar(n), n, h.options());

        // 3. Gini = (Σ (2i-n-1) * v_i) / (n * Σ v_i)
        Tensor numerator = mul(sub(mul(index, new Scalar(2)), new Scalar(n + 1)), v).sum();
        Tensor denominator = mul(v.sum(), new Scalar(n));

        return div(numerator, denominator);
    }
}