package org.bytedeco.pytorch.geometric.model;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import java.text.DecimalFormat;

import static org.bytedeco.pytorch.global.torch.*;

public class FraudDetectionTrain {

    public static void main(String[] args) {
        System.out.println("=== 初始化金融反欺诈训练任务 ===");

        // 1. 配置超参数
        long numNodes = 1000;      // 账户数量
        long featureDim = 32;      // 特征维度 (如：交易额、频率、时间差等)
        long hiddenDim = 64;       // 隐层维度
        long numEdges = 3000;      // 交易记录数量
        double learningRate = 0.01;
        int epochs = 300;

        // 2. 生成假数据 (Dummy Data)
        System.out.println("正在生成模拟交易图数据...");

        // 节点特征 X [1000, 32]
        Tensor x = torch.randn(new long[]{numNodes, featureDim});

        // 边 edge_index [2, 3000] (随机连接)
        Tensor edge_index = torch.randint(numNodes, new long[]{2, numEdges}, x.options().dtype(new ScalarTypeOptional(ScalarType.Long)));

        // 标签 Y [1000, 1] (0: 正常, 1: 欺诈)
        // 假设 10% 是欺诈
        Tensor y = torch.rand(new long[]{numNodes, 1}); // 生成 0-1 之间的数
        y = y.lt(new Scalar(0.1)).to(ScalarType.Float); // 小于 0.1 的设为 1.0 (Float类型用于BCE Loss)

        System.out.println("数据准备完毕. Nodes: " + numNodes + ", Edges: " + numEdges);
        System.out.println("欺诈样本数: " + y.sum().item().toFloat());

        // 3. 实例化模型
        FraudGNNModel model = new FraudGNNModel(featureDim, hiddenDim, 1, 0.5); // output dim 1 for binary
        model.train(true); // 开启训练模式 (启用 Dropout)

//        model.to(new Device(DeviceType.CUDA));
        // 4. 定义优化器 (Adam)
        AdamOptions options = new AdamOptions(learningRate);
        Adam optimizer = new Adam(model.parameters(), options);

        // 5. 训练循环
        System.out.println("\n--- 开始训练 ---");
        DecimalFormat df = new DecimalFormat("0.0000");

        for (int epoch = 1; epoch <= epochs; epoch++) {
            // A. 梯度清零
            optimizer.zero_grad();

            // B. 前向传播
            Tensor logits = model.forward(x, edge_index);

            // C. 计算损失 (Binary Cross Entropy with Logits)
            // 这种 Loss 函数自带 Sigmoid，数值更稳定
            Tensor loss = torch.binary_cross_entropy_with_logits(logits, y);

            // D. 反向传播
            loss.backward();

            // E. 参数更新
            optimizer.step();

            // F. 监控进度 (每10轮打印一次)
            if (epoch % 10 == 0) {
                float lossVal = loss.item().toFloat();
                float acc = computeAccuracy(logits, y);
                System.out.println("Epoch [" + epoch + "/" + epochs + "] " +
                        "Loss: " + df.format(lossVal) + " | Acc: " + df.format(acc * 100) + "%");
            }
        }

        // 6. 简单的推理测试
        System.out.println("\n--- 推理演示 (前5个用户) ---");
        model.eval(); // 关闭 Dropout
        Tensor finalLogits = model.forward(x, edge_index);
        Tensor probs = torch.sigmoid(finalLogits); // 转换为 0-1 概率

        for (long i = 0; i < 5; i++) {
            float prob = probs.select(0, i).item().toFloat();
            float label = y.select(0, i).item().toFloat();
            String status = prob > 0.5 ? "RISK" : "SAFE";
            System.out.printf("User %d: Prob=%.4f, Label=%.0f -> %s%n", i, prob, label, status);
        }
    }

    // 辅助方法：计算准确率
    private static float computeAccuracy(Tensor logits, Tensor targets) {
        // Sigmoid -> >0.5 ? 1 : 0
        Tensor probs = torch.sigmoid(logits);
        Tensor preds = probs.gt(new Scalar(0.5)).to(ScalarType.Float);
        Tensor correct = preds.eq(targets).to(ScalarType.Float);
        return correct.mean().item().toFloat();
    }
}