package org.bytedeco.pytorch.rl;

import org.bytedeco.pytorch.rl.critic.ActorCriticNetwork;
import org.bytedeco.pytorch.rl.sampler.ParallelSampler;
import org.bytedeco.pytorch.rl.trainer.GRPOTrainer;

public class RLSystemIntegrator {
    public static void main(String[] args) {
        System.out.println("=== 启动强化学习系统集成验证 ===");

        // 1. 初始化
        ActorCriticNetwork model = new ActorCriticNetwork(4, 2);
        ReplayBuffer buffer = new ReplayBuffer();
        GRPOTrainer trainer = new GRPOTrainer(model);
        ParallelSampler sampler = new ParallelSampler(model, 4); // 4 线程并行

        // 2. 迭代循环
        for (int iter = 1; iter <= 3; iter++) {
            System.out.println("迭代周期: " + iter);

            // 采样
            sampler.collectTrajectories(buffer);
            System.out.println("采样完成，当前 Buffer 大小: " + buffer.size());

            // 训练
            trainer.trainBatch(buffer);
            System.out.println("训练步完成，策略已更新。");

            // 清理
            buffer.clear();
        }

        System.out.println("=== [SUCCESS] 系统集成测试通过 ===");
    }
}