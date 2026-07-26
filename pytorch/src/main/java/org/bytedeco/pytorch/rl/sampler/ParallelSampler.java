package org.bytedeco.pytorch.rl.sampler;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.rl.critic.ActorCriticNetwork;
import org.bytedeco.pytorch.rl.ReplayBuffer;

import java.util.stream.IntStream;

import static org.bytedeco.pytorch.global.torch.*;
public class ParallelSampler {
    private final ActorCriticNetwork globalModel;
    private final int numWorkers;

    public ParallelSampler(ActorCriticNetwork model, int numWorkers) {
        this.globalModel = model;
        this.numWorkers = numWorkers;
    }

    public void collectTrajectories(ReplayBuffer sharedBuffer) {
        // 使用并行流或线程池同时在多个环境采集
        IntStream.range(0, numWorkers).parallel().forEach(id -> {
//            try (PointerScope scope = new PointerScope()) {
                // 1. 获取当前状态 (模拟环境)
                Tensor state = randn(new long[]{1, 4}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

                // 2. 预测并采样
                // 注意：多线程下 forward 需要同步或使用独立副本
                synchronized(globalModel) {
                    Distribution dist = globalModel.forward_policy(state);
                    Tensor action = dist.sample();
                    Tensor logProb = dist.log_prob(action).sum(-1);

                    var r = scalar_tensor(new Scalar(1.0), new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
                    // 3. 存储到 Buffer
                    sharedBuffer.push(state, action, logProb,r,r );
                }
//            }
        });
    }
}