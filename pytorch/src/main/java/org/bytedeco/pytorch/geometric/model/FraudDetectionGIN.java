package org.bytedeco.pytorch.geometric.model;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.geometric.nn.model.GIN;

import java.text.DecimalFormat;

import static org.bytedeco.pytorch.global.torch.ScalarType;

public class FraudDetectionGIN {

    public static void main(String[] args) {
        System.out.println("=== org.bytedeco.pytorch.geometric.nn.model.GIN 金融反欺诈训练任务 (Graph Isomorphism Network) ===");

        // --- 1. 设备配置 (优先使用 CUDA) ---
//        Device device;
////        if (torch.hasMPS()) then: device = new Device("mps") else  device = new Device("cpu");
//        if (torch_cuda.is_available()) {
//            device = new Device("cuda");
//            System.out.println("Using CUDA GPU.");
//        } else {
//            device = new Device("cpu");
//            System.out.println("Using CPU.");
//        }

        if (!torch.hasCUDA()) {
            System.out.println("===(CUDA) not support ===");
//            throw new RuntimeException("Need CUDA for this enterprise demo!");
        }
        Device device = new Device("mps");// new Device("cuda");

        // --- 2. 超参数 ---
        long numNodes = 2000;
        long featureDim = 64;
        long hiddenDim = 64; // org.bytedeco.pytorch.geometric.nn.model.GIN 通常需要较大的隐层宽度来发挥 MLP 的威力
        long outChannels = 1; // 二分类 (Logit)
        int numLayers = 4;    // 4层 org.bytedeco.pytorch.geometric.nn.model.GIN
        long numEdges = 10000;
        double lr = 0.005;
        int epochs = 1500;

        // --- 3. 生成数据 (并移动到设备) ---
        System.out.println("生成模拟数据...");
        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(device));
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(device));

        // 特征 X
        Tensor x = torch.randn(new long[]{numNodes, featureDim}, floatOpts);

        // 边 edge_index (必须是 Long)
        Tensor edge_index = torch.randint(numNodes, new long[]{2, numEdges}, longOpts);

        // 标签 Y (0: 正常, 1: 欺诈)
        Tensor y = torch.rand(new long[]{numNodes, 1}, floatOpts);
        y = y.lt(new Scalar(0.1)).to(ScalarType.Float); // 10% 欺诈率

        // --- 4. 初始化模型 ---
        // 注意：outChannels=1 用于 BCEWithLogitsLoss
        GIN model = new GIN(featureDim, hiddenDim, outChannels, numLayers, 0.5);
        model.to(device,true);
        model.train(true);

        // --- 5. 优化器 ---
        AdamOptions optimOpts = new AdamOptions(lr);
        // org.bytedeco.pytorch.geometric.nn.model.GIN 包含 BatchNorm，有时需要配合 Weight Decay，这里暂时设为 0
        optimOpts.weight_decay().put(5e-4);
        Adam optimizer = new Adam(model.parameters(), optimOpts);

        // --- 6. 训练循环 ---
        System.out.println("\n--- 开始训练 org.bytedeco.pytorch.geometric.nn.model.GIN ---");
        DecimalFormat df = new DecimalFormat("0.0000");

        for (int epoch = 1; epoch <= epochs; epoch++) {
            try (PointerScope scope = new PointerScope()) {
                optimizer.zero_grad();

                // Forward
                Tensor logits = model.forward(x, edge_index);

                // Loss (BCE With Logits)
                Tensor loss = torch.binary_cross_entropy_with_logits(logits, y);

                // Backward
                loss.backward();
                optimizer.step();

                // Monitoring
                if (epoch % 10 == 0) {
                    float lossVal = loss.item().toFloat();
                    float acc = computeAccuracy(logits, y);
                    System.out.println("Epoch [" + epoch + "/" + epochs + "] " +
                            "Loss: " + df.format(lossVal) + " | Acc: " + df.format(acc * 100) + "%");
                }
            }

        }

        // --- 7. 推理评估 ---
        evaluate(model, x, edge_index, y);
    }

    private static float computeAccuracy(Tensor logits, Tensor targets) {
        Tensor probs = torch.sigmoid(logits);
        Tensor preds = probs.gt(new Scalar(0.5)).to(ScalarType.Float);
        Tensor correct = preds.eq(targets).to(ScalarType.Float);
        return correct.mean().item().toFloat();
    }

    private static void evaluate(GIN model, Tensor x, Tensor edge_index, Tensor y) {
        System.out.println("\n--- 最终模型评估 ---");
        model.eval(); // 切换到评估模式 (关闭 BN 和 Dropout)

        Tensor logits = model.forward(x, edge_index);
        Tensor probs = torch.sigmoid(logits);

        // 简单打印前5个结果 (需转回 CPU)
        Tensor probsCpu = probs.cpu();
        Tensor yCpu = y.cpu();

        int fraudDetected = 0;
        for(long i=0; i<probsCpu.size(0); i++) {
            if (probsCpu.select(0, i).item().toFloat() > 0.5) {
                fraudDetected++;
            }
        }
        System.out.println("总预测欺诈用户数: " + fraudDetected);

        for (long i = 0; i < 5; i++) {
            float p = probsCpu.select(0, i).item().toFloat();
            float label = yCpu.select(0, i).item().toFloat();
            System.out.printf("User %d: Score=%.4f, TrueLabel=%.0f%n", i, p, label);
        }
    }
}