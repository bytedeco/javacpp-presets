package org.bytedeco.pytorch.geometric.model;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;

import java.text.DecimalFormat;

import static org.bytedeco.pytorch.global.torch.ScalarType;

public class RecSysTraining {

    public static void main(String[] args) {
        // --- 0. CUDA 环境检查 ---
//        if (!torch_cuda.is_available()) {
//            System.out.println("=== 推荐系统训练启动 (CUDA) not support ===");
////            throw new RuntimeException("Need CUDA for this enterprise demo!");
//        }
        if (!torch.hasCUDA()) {
            System.out.println("=== 推荐系统训练启动 (CUDA) not support ===");
//            throw new RuntimeException("Need CUDA for this enterprise demo!");
        }
        Device device = new Device("cpu");// new Device("mps") ;// new Device("cuda");
        System.out.println("=== 推荐系统训练启动 (CUDA) ===  mac is have mps gpu" + torch.hasMPS());

        try {
            // 任务 1: 电商召回
            trainEcommerceRecall(device);

            System.out.println("\n--------------------------------------\n");

            // 任务 2: 社交排序
            trainSocialRanking(device);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * ===================================================================
     * 场景 1: 电商召回训练 (Implicit Feedback, Negative Sampling)
     * ===================================================================
     */
    private static void trainEcommerceRecall(Device device) {
        System.out.println(">>> 正在启动 [电商商品召回] 模型训练...");

        // 1. 模拟异构图 (为了简化 GNN 实现，我们将 User 和 Item 放在同一个图中)
        // ID 0~999: Users, ID 1000~2999: Items
        long numUsers = 10000;
        long numItems = 20000;
        long totalNodes = numUsers + numItems;
        long featureDim = 64;
        long numInteractions = 20000; // 购买记录

        // 2. 数据生成 (GPU)
        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(device));
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(device));

        Tensor x = torch.randn(new long[]{totalNodes, featureDim}, floatOpts);

        // 构造 User -> Item 的边 (只存正样本)
        Tensor userIdx = torch.randint(numUsers, new long[]{numInteractions}, longOpts);
        Tensor itemIdx = torch.randint(numItems, new long[]{numInteractions}, longOpts).add(new Scalar(numUsers)); // Offset Item IDs

        // edge_index: [2, E]
        Tensor edge_index = torch.stack(new TensorVector(userIdx, itemIdx), 0);

        // 3. 模型初始化
        RecSysModels.EcommerceRecallModel model = new RecSysModels.EcommerceRecallModel(featureDim, 64, 32);
        model.to(device, ScalarType.Float, true);
        model.train(true);

        Adam optimizer = new Adam(model.parameters(), new AdamOptions(0.01));
        DecimalFormat df = new DecimalFormat("0.0000");

        // 4. 训练循环 (Link Prediction style)
        for (int epoch = 1; epoch <= 1000; epoch++) {
            try (PointerScope scope = new PointerScope()) {
                optimizer.zero_grad();

                // A. 前向传播：获取所有节点的 Embedding
                Tensor z = model.forward(x, edge_index);

                // B. 获取正样本对的 Embedding
                // edge_index[0] 是 users, edge_index[1] 是 items
                Tensor posUserEmbed = z.index_select(0, edge_index.select(0, 0));
                Tensor posItemEmbed = z.index_select(0, edge_index.select(0, 1));

                // C. 负采样 (Negative Sampling)
                // 随机生成同样数量的 Item ID 作为负样本
                Tensor negItemIds = torch.randint(numItems, new long[]{numInteractions}, longOpts).add(new Scalar(numUsers));
                Tensor negItemEmbed = z.index_select(0, negItemIds);

                // D. 计算相似度得分 (Dot Product)
                Tensor posScores = model.predictSimilarity(posUserEmbed, posItemEmbed);
                Tensor negScores = model.predictSimilarity(posUserEmbed, negItemEmbed);

//                System.out.println("recall ing...");
                // E. 计算 Margin Loss (或者 BPR Loss)
                // Loss = ReLU(1 - pos + neg) + epsilon
                // 目标: pos 分数越高越好，neg 分数越低越好
                // 1. 计算 (neg - pos)
                Tensor diff = negScores.sub(posScores);

// 2. 计算 (diff + 1.0) -> ReLU -> Mean
                Tensor loss = torch.relu(diff.add(new Scalar(1.0))).mean();
//                Tensor margin = new Tensor(new Scalar(1.0)).to(device, torch.ScalarType.Float); //big bug
//                System.out.println("recall ing2...");
//                Tensor loss = torch.relu(margin.sub(posScores).add(negScores)).mean();
//                System.out.println("recall ing3...");
                loss.backward();
                optimizer.step();

                if (epoch % 20 == 0) {
                    System.out.println("Recall Epoch [" + epoch + "] Loss: " + df.format(loss.item().toFloat()));
                }
            }
        }
        System.out.println("电商召回模型训练完成。");
    }

    /**
     * ===================================================================
     * 场景 2: 社交相亲排序训练 (Binary Classification)
     * ===================================================================
     */
    private static void trainSocialRanking(Device device) {
        System.out.println(">>> 正在启动 [社交相亲排序] 模型训练...");

        // 1. 模拟数据
        long numUsers = 1000;
        long featureDim = 32;
        long numFriendships = 5000; // 现有的社交关系图 (用于 GNN 聚合)
        long numCandidates = 2000;  // 待预测的相亲对 (用于 MLP 排序)

        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(device));
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(device));

        // 用户画像
        Tensor x = torch.randn(new long[]{numUsers, featureDim}, floatOpts);

        // 社交关系图 (GNN 用的边)
        Tensor edge_index = torch.randint(numUsers, new long[]{2, numFriendships}, longOpts);

        // 待打分的候选对 (UserA, UserB)
        Tensor srcUsers = torch.randint(numUsers, new long[]{numCandidates}, longOpts);
        Tensor dstUsers = torch.randint(numUsers, new long[]{numCandidates}, longOpts);

        // 真实标签 (1: 匹配成功, 0: 匹配失败)
        Tensor labels = torch.rand(new long[]{numCandidates, 1}, floatOpts).gt(new Scalar(0.7)).to(ScalarType.Float);

        // 2. 模型初始化
        RecSysModels.SocialRankingModel model = new RecSysModels.SocialRankingModel(featureDim, 32, 16, 4); // 4 heads
        model.to(device, ScalarType.Float,true);
        model.train(true);

        Adam optimizer = new Adam(model.parameters(), new AdamOptions(0.005));
        DecimalFormat df = new DecimalFormat("0.0000");

        // 3. 训练循环
        for (int epoch = 1; epoch <= 100; epoch++) {
            try (PointerScope scope = new PointerScope()) {
                optimizer.zero_grad();

                // A. 前向传播
                // 输入：全图信息 + 待预测的候选对
                Tensor logits = model.forward(x, edge_index, srcUsers, dstUsers);

                // B. Loss (BCEWithLogits)
                Tensor loss = torch.binary_cross_entropy_with_logits(logits, labels);

                loss.backward();
                optimizer.step();

                if (epoch % 20 == 0) {
                    // 计算准确率
                    Tensor probs = torch.sigmoid(logits);
                    Tensor preds = probs.gt(new Scalar(0.5)).to(ScalarType.Float);
                    float acc = preds.eq(labels).to(ScalarType.Float).mean().item().toFloat();

                    System.out.println("Ranking Epoch [" + epoch + "] Loss: " + df.format(loss.item().toFloat())
                            + " | Acc: " + df.format(acc * 100) + "%");
                }
            }
        }
        System.out.println("社交排序模型训练完成。");
    }
}
