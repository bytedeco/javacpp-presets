package org.bytedeco.pytorch.rl.sampler;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.rl.critic.ActorCriticNetwork;
import org.bytedeco.pytorch.rl.ReplayBuffer;

import static org.bytedeco.pytorch.global.torch.*;
public class GRPOSampler {
    /**
     * @param groupSize 组大小 (G)
     * @return Tensor[] {states, actions, oldLogProbs, groupRewards}
     */
    public void collectGroupData(ActorCriticNetwork net, Tensor state, int groupSize, ReplayBuffer buffer) {
        try (PointerScope scope = new PointerScope()) {
            // 将单个 state 扩展为 groupSize 份
            Tensor expandedState = state.expand(new long[]{groupSize, state.size(-1)}, false);

            // 一次性前向传播得到分布
            Distribution dist = net.forward_policy(expandedState);
            Tensor actions = dist.sample();
            Tensor logProbs = dist.log_prob(actions).sum(-1);

            // 模拟环境对这一组动作的打分 (例如：回答的质量、计算的准确度)
            Tensor rewards = randn(new long[]{groupSize}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

            // 存入 Buffer。注意：这里我们将整组数据作为一个 Batch 存入
            buffer.push(expandedState, actions, logProbs,zeros_like(logProbs), rewards);
        }
    }
}