package org.bytedeco.pytorch.rl.agent;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Distribution;

import static org.bytedeco.pytorch.global.torch.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.ActorCritic;
import org.bytedeco.pytorch.rl.critic.CartPoleActorCritic;


import static org.bytedeco.pytorch.global.torch.*;

/**
 * <b>Guided-Reward PPO</b> (historically mislabeled "GRPO" in this package).
 *
 * <p><b>Naming note — do not confuse with DeepSeek GRPO:</b>
 * <ul>
 *   <li><b>This class</b> = env reward + auxiliary <em>guide</em> reward, fused then
 *       optimized with a PPO clipped surrogate (classic actor-critic).</li>
 *   <li><b>DeepSeek / HF-TRL GRPO</b> = <em>Group-Relative</em> Policy Optimization:
 *       several completions per prompt, advantages from in-group reward
 *       normalization, often critic-free. See
 *       {@link org.bytedeco.pytorch.rl.agent.GroupRelativePPOAgent},
 *       {@link org.bytedeco.pytorch.rl.loss.GRPOLoss},
 *       {@link org.bytedeco.pytorch.llm.trl.GRPOTrainer}.</li>
 * </ul>
 *
 * <p>Prefer the explicit alias {@link GuidedRewardPPOAgent} in new code.
 * {@code GRPOAgent} is kept as a binary-compatible name.
 */
public class GRPOAgent extends AbstractRLAgent {
    // Guided-reward PPO hyper-parameters
    private final float clipEps;          // PPO clip ε (default 0.2)
    private final float gamma;            // discount
    private final float gaeLambda;        // GAE-λ
    private final float guideRewardWeight;// weight on guide reward in the fuse
    private final float entropyCoeff;     // entropy bonus

    // ===================== 标准构造函数（推荐，符合 Java 语法）=====================
    /**
     * 标准构造函数：自定义所有核心组件和超参数
     * @param model 策略-价值模型（任意 AbstractActorCritic 子类）
     * @param optimizer 优化器（Adam/SGD）
     * @param replayBuffer 经验缓冲区（存储状态/动作/奖励/引导奖励等）
     * @param clipEps PPO 裁剪系数
     * @param gamma 折扣因子
     * @param gaeLambda GAE 系数
     * @param guideRewardWeight 引导奖励权重
     * @param entropyCoeff 熵正则化系数
     */
    public GRPOAgent(AbstractActorCritic model,
                     Optimizer optimizer,
                     ReplayBuffer replayBuffer,
                     float clipEps,
                     float gamma,
                     float gaeLambda,
                     float guideRewardWeight,
                     float entropyCoeff) {
        // 1. super() 必须是第一条语句，注入核心组件
        super(model, optimizer, replayBuffer);
        // 2. 初始化 GRPO 超参数
        this.clipEps = clipEps;
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.guideRewardWeight = guideRewardWeight;
        this.entropyCoeff = entropyCoeff;
    }

    // ===================== 简化构造函数（快速创建，默认超参数）=====================
    /**
     * 简化构造函数：默认超参数，仅需指定状态/动作维度和动作空间类型
     * @param stateDim 状态维度
     * @param actionDim 动作维度
     * @param actionSpaceType 动作空间类型（DISCRETE/CONTINUOUS）
     */
    /**
     * 简化构造：同一 model 绑定 optimizer（不再 close 临时模型）。
     */
    public GRPOAgent(long stateDim, long actionDim, AbstractActorCritic.ActionSpaceType actionSpaceType) {
        this(build(stateDim, actionDim, actionSpaceType, 3e-4f),
                0.2f, 0.99f, 0.95f, 0.5f, 0.01f);
    }

    public static GRPOAgent create(long stateDim, long actionDim,
                                   AbstractActorCritic.ActionSpaceType actionSpaceType) {
        return new GRPOAgent(stateDim, actionDim, actionSpaceType);
    }

    private GRPOAgent(Object[] built, float clipEps, float gamma, float gaeLambda,
                      float guideRewardWeight, float entropyCoeff) {
        super((AbstractActorCritic) built[0], (Optimizer) built[1], new ReplayBuffer());
        this.clipEps = clipEps;
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.guideRewardWeight = guideRewardWeight;
        this.entropyCoeff = entropyCoeff;
    }

    private static Object[] build(long stateDim, long actionDim,
                                  AbstractActorCritic.ActionSpaceType actionSpaceType, float lr) {
        AbstractActorCritic model = createActorCriticModel(stateDim, actionDim, actionSpaceType);
        AdamOptions optOptions = new AdamOptions();
        optOptions.lr().put(lr);
        return new Object[]{model, new Adam(model.parameters(), optOptions)};
    }

    private static AbstractActorCritic createActorCriticModel(long stateDim, long actionDim,
                                                              AbstractActorCritic.ActionSpaceType actionSpaceType) {
        if (actionSpaceType == AbstractActorCritic.ActionSpaceType.CONTINUOUS) {
            return new ActorCritic(stateDim, actionDim);
        }
        return new CartPoleActorCritic(stateDim, actionDim);
    }

    // ===================== 实现抽象方法（AbstractRLAgent 核心要求）=====================
    /**
     * 核心训练方法：实现 AbstractRLAgent 的 trainStep() 抽象方法
     * 逻辑：1. 从缓冲区读取数据；2. 融合环境奖励和引导奖励；3. 计算 GAE 优势；4. PPO 裁剪优化
     * @return GRPO 总损失值
     */
    @Override
    public Tensor trainStep() {
        // 1. 校验缓冲区数据
        ReplayBuffer buffer = super.getReplayBuffer();
        if (buffer.size() == 0) {
            throw new IllegalStateException("GRPO 经验缓冲区为空，无法训练！");
        }

        // 2. 从缓冲区读取核心数据（GRPO 需额外读取引导奖励）
        Tensor states = buffer.getStates();
        Tensor actions = buffer.getActions();
        Tensor oldLogProbs = buffer.getOldLogProbs(); // 旧策略对数概率
        Tensor envRewards = buffer.getEnvRewards();   // 环境奖励
        Tensor guideRewards = buffer.getGuideRewards(); // 引导奖励
        Tensor dones = buffer.getDones();             // 结束标志
        Tensor values = buffer.getValues();           // 价值预测

        // 3. 融合环境奖励和引导奖励（核心：加权求和）
        Tensor fusedRewards = envRewards.mul(new Scalar(1 - guideRewardWeight))
                .add(guideRewards.mul(new Scalar(guideRewardWeight)));

        // 4. 计算 GAE 优势和折扣回报
        Tensor[] gaeResult = computeGAE(fusedRewards, values, dones, gamma, gaeLambda);
        Tensor advantages = gaeResult[0];
        Tensor returns = gaeResult[1];

        // 5. 标准化优势（提升训练稳定性）
        advantages = (advantages.sub(advantages.mean())).div(advantages.std().add(new Scalar(1e-8)));

        // 6. 获取当前策略分布，计算新对数概率
        AbstractActorCritic model = super.getModel();
        Distribution dist = model.getDistribution(states);
        Tensor currentLogProbs = sumActionLogProb(dist.log_prob(actions));

        // 7. 计算 PPO 裁剪损失（GRPO 核心保留 PPO 裁剪机制）
        Tensor ratio = exp(currentLogProbs.sub(oldLogProbs));
        Tensor surr1 = ratio.mul(advantages.detach());
        Tensor surr2 = clamp(ratio,
                new ScalarOptional(new Scalar(1.0 - clipEps)),
                new ScalarOptional(new Scalar(1.0 + clipEps))).mul(advantages.detach());
        Tensor actorLoss = min(surr1, surr2).mean().neg();

        // 8. 计算 Critic 损失（价值函数拟合融合奖励的回报）
        Tensor currentValues = model.getValue(states).squeeze(-1);
        Tensor criticLoss = mse_loss(currentValues, returns);

        // 9. 熵正则化（鼓励探索）
        Tensor entropy = dist.entropy().mean();
        Tensor entropyBonus = entropy.mul(new Scalar(entropyCoeff));

        // 10. 总损失 = 策略损失 + 价值损失 - 熵奖励
        Tensor totalLoss = actorLoss.add(criticLoss.mul(new Scalar(0.5)))
                .sub(entropyBonus);

        // 11. 反向传播 + 优化器更新
        Optimizer optimizer = super.optimizer;
        optimizer.zero_grad();
        totalLoss.backward();
        clip_grad_norm_(model.parameters(), 0.5); // 梯度裁剪
        optimizer.step();

        // 12. 释放临时张量（避免 Native 内存泄漏）
        fusedRewards.close();
        advantages.close();
        returns.close();
        currentLogProbs.close();
        ratio.close();
        surr1.close();
        surr2.close();
        actorLoss.close();
        currentValues.close();
        criticLoss.close();
        entropy.close();
        entropyBonus.close();

        // 13. 返回总损失（detach 避免梯度传播）
        return totalLoss.detach();
    }

    /**
     * 采样方法：实现 AbstractRLAgent 的 sample() 抽象方法
     * 逻辑：输入状态 → 采样动作 → 返回动作/对数概率/价值（用于收集经验）
     * @param state 状态张量 [batch, stateDim]
     * @return 数组：[动作, 对数概率, 价值]
     */
    @Override
    public Tensor[] sample(Tensor state) {
        AbstractActorCritic model = super.getModel();
        model.train(true); // 切换到训练模式

        // 1. 获取动作分布并采样
        Distribution dist = model.getDistribution(state);
        Tensor action = dist.sample();

        // 2. 计算动作的对数概率（用于后续 PPO 裁剪）
        Tensor logProb = sumActionLogProb(dist.log_prob(action));

        // 3. 计算状态价值（用于 GAE 优势计算）
        Tensor value = model.getValue(state);

        // 4. 返回采样结果（detach 避免梯度图污染）
        return new Tensor[]{
                action.detach().clone(),
                logProb.detach().clone(),
                value.detach().clone()
        };
    }

    // ===================== GRPO 核心扩展方法 =====================
    /**
     * 计算 GAE (Generalized Advantage Estimation) 优势
     * GRPO 核心：基于融合奖励计算优势
     */
    /**
     * @param rewards [T] or [T,B]
     * @param values  [T] or [T+1] — if length T, bootstrap V(s_T)=0
     * @param dones   1 = episode ended at t
     */
    public Tensor[] computeGAE(Tensor rewards, Tensor values, Tensor dones, float gamma, float tau) {
        long T = rewards.size(0);
        long valueLen = values.size(0);
        if (valueLen != T && valueLen != T + 1) {
            throw new IllegalArgumentException(
                    "values length must be T or T+1 (got " + valueLen + ", T=" + T + ")");
        }
        Tensor advantages = zeros_like(rewards);
        Tensor gae = scalar_tensor(new Scalar(0.0), rewards.options());

        for (long t = T - 1; t >= 0; t--) {
            Tensor nextVal;
            if (valueLen == T + 1) {
                nextVal = values.select(0, t + 1);
            } else if (t == T - 1) {
                nextVal = zeros_like(values.select(0, t));
            } else {
                nextVal = values.select(0, t + 1);
            }
            Tensor mask = dones.select(0, t).neg().add(new Scalar(1.0)); // 1 - done
            Tensor delta = rewards.select(0, t)
                    .add(nextVal.mul(new Scalar(gamma)).mul(mask))
                    .sub(values.select(0, t));
            gae = delta.add(gae.mul(new Scalar(gamma * tau)).mul(mask));
            advantages.select(0, t).copy_(gae);
        }

        Tensor valueSlice = valueLen == T + 1
                ? values.slice(0, new LongOptional(0), new LongOptional(T), 1)
                : values;
        Tensor returns = advantages.add(valueSlice);
        return new Tensor[]{advantages, returns};
    }

    /**
     * 手动更新方法（兼容旧代码）：直接传入融合奖励数据训练
     */
    public Tensor update(Tensor states, Tensor actions, Tensor oldLogProbs,
                         Tensor fusedRewards, Tensor dones, Tensor values) {
        // 计算 GAE 优势和回报
        Tensor[] gaeResult = computeGAE(fusedRewards, values, dones, gamma, gaeLambda);
        Tensor advantages = gaeResult[0];
        Tensor returns = gaeResult[1];

        // 复用 trainStep 核心逻辑
        AbstractActorCritic model = super.getModel();
        Distribution dist = model.getDistribution(states);
        Tensor currentLogProbs = sumActionLogProb(dist.log_prob(actions));

        Tensor ratio = exp(currentLogProbs.sub(oldLogProbs));
        Tensor surr1 = ratio.mul(advantages.detach());
        Tensor surr2 = clamp(ratio,new ScalarOptional(new Scalar( 1 - clipEps)), new ScalarOptional(new Scalar(1 + clipEps))).mul(advantages.detach());
        Tensor actorLoss = min(surr1, surr2).mean().neg();

        Tensor currentValues = model.getValue(states).squeeze(-1);
        Tensor criticLoss = mse_loss(currentValues, returns);
        Tensor entropy = dist.entropy().mean();

        Tensor totalLoss = actorLoss.add(criticLoss.mul(new Scalar(0.5))).sub(entropy.mul(new Scalar(entropyCoeff)));

        super.optimizer.zero_grad();
        totalLoss.backward();
        clip_grad_norm_(model.parameters(), 0.5);
        super.optimizer.step();

        // 释放临时张量
        advantages.close();
        returns.close();
        currentLogProbs.close();
        ratio.close();
        surr1.close();
        surr2.close();
        actorLoss.close();
        currentValues.close();
        criticLoss.close();
        entropy.close();

        return totalLoss.detach();
    }

    // ===================== Getter/Setter（方便调参）=====================
    public float getClipEps() {
        return clipEps;
    }

    public float getGamma() {
        return gamma;
    }

    public float getGaeLambda() {
        return gaeLambda;
    }

    public float getGuideRewardWeight() {
        return guideRewardWeight;
    }

    public float getEntropyCoeff() {
        return entropyCoeff;
    }

    /** Discrete Categorical log_prob has no action dim — only sum multi-dim continuous actions. */
    private static Tensor sumActionLogProb(Tensor logProb) {
        if (logProb.dim() <= 1) {
            return logProb;
        }
        return logProb.sum(-1);
    }

    @Override
    public void close() {
        super.close();
    }
}
