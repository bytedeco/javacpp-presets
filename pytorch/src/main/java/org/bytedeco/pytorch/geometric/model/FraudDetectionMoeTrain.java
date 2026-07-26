package org.bytedeco.pytorch.geometric.model;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
//import org.gnn.demo.model.org.bytedeco.pytorch.geometric.model.FraudGNNModel;

import java.text.DecimalFormat;

import static org.bytedeco.pytorch.global.torch.*;

public class FraudDetectionMoeTrain {

    public static void main(String[] args) {
        System.out.println("=== 初始化金融反欺诈训练任务 ===");

        // 1. 配置超参数
        long numNodes = 1000;
        long featureDim = 32;
        long hiddenDim = 64;
        long numEdges = 3000;
        double learningRate = 0.01;
        int epochs = 100;

        // 2. 生成假数据 (Dummy Data)
        System.out.println("正在生成模拟交易图数据...");

        // 节点特征 X [1000, 32] (默认是 Float)
        Tensor x = torch.randn(new long[]{numNodes, featureDim});

        // ---------------------------------------------------------------------
        // [FIX START] 修复 edge_index 类型问题
        // ---------------------------------------------------------------------
        // 务必确保 edge_index 是 Long 类型 (Int64)，否则 index_select 会报错
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
        Tensor edge_index = torch.randint(numNodes, new long[]{2, numEdges}, longOpts);

        // 双重保险：强制转为 Long，防止 randint 意外返回 Float
        edge_index = edge_index.to(ScalarType.Long);
        // ---------------------------------------------------------------------
        // [FIX END]
        // ---------------------------------------------------------------------

        // 标签 Y [1000, 1]
        Tensor y = torch.rand(new long[]{numNodes, 1});
        y = y.lt(new Scalar(0.1)).to(ScalarType.Float); // Float 类型用于 BCE Loss

        System.out.println("数据准备完毕. Nodes: " + numNodes + ", Edges: " + numEdges);
        System.out.println("欺诈样本数: " + y.sum().item().toFloat());

        // 3. 实例化模型
        FraudGNNModel model = new FraudGNNModel(featureDim, hiddenDim, 1, 0.5);
        model.train(true);

        // 4. 定义优化器
        AdamOptions options = new AdamOptions(learningRate);
        Adam optimizer = new Adam(model.parameters(), options);

        // 5. 训练循环
        System.out.println("\n--- 开始训练 ---");
        DecimalFormat df = new DecimalFormat("0.0000");

        for (int epoch = 1; epoch <= epochs; epoch++) {
            optimizer.zero_grad();

            // Forward
            Tensor logits = model.forward(x, edge_index);

            // Loss
            Tensor loss = torch.binary_cross_entropy_with_logits(logits, y);

            // Backward
            loss.backward();
            optimizer.step();

            // Log
            if (epoch % 10 == 0) {
                float lossVal = loss.item().toFloat();
                float acc = computeAccuracy(logits, y);
                System.out.println("Epoch [" + epoch + "/" + epochs + "] " +
                        "Loss: " + df.format(lossVal) + " | Acc: " + df.format(acc * 100) + "%");
            }
        }

        // 6. 推理演示
        System.out.println("\n--- 推理演示 (前5个用户) ---");
        model.eval();
        Tensor finalLogits = model.forward(x, edge_index);
        Tensor probs = torch.sigmoid(finalLogits);

        for (long i = 0; i < 5; i++) {
            float prob = probs.select(0, i).item().toFloat();
            float label = y.select(0, i).item().toFloat();
            String status = prob > 0.5 ? "RISK" : "SAFE";
            System.out.printf("User %d: Prob=%.4f, Label=%.0f -> %s%n", i, prob, label, status);
        }
    }

    private static float computeAccuracy(Tensor logits, Tensor targets) {
        Tensor probs = torch.sigmoid(logits);
        Tensor preds = probs.gt(new Scalar(0.5)).to(ScalarType.Float);
        Tensor correct = preds.eq(targets).to(ScalarType.Float);
        return correct.mean().item().toFloat();
    }
}
