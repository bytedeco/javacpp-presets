package org.bytedeco.pytorch.rl.agent;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;

import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.ActorCritic;
import static org.bytedeco.pytorch.global.torch.mse_loss;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.CartPoleActorCritic;

import static org.bytedeco.pytorch.global.torch.mse_loss;

/**
 * A2C (Advantage Actor-Critic) 算法实现
 * 修正：super() 必须作为构造函数第一条语句
 */
public class A2CAgent extends AbstractRLAgent {
    // A2C 超参数（可通过构造函数配置）
    private final float entropyCoeff; // 熵正则化系数（鼓励探索）
    private final float gamma;         // 折扣因子

    // ===================== 标准构造函数（推荐）=====================
    /**
     * 标准构造函数：兼容抽象父类，支持自定义所有核心组件
     * @param model 策略-价值模型
     * @param optimizer 优化器
     * @param replayBuffer 经验缓冲区
     * @param gamma 折扣因子（如 0.99）
     * @param entropyCoeff 熵正则化系数（如 0.01）
     */
    public A2CAgent(AbstractActorCritic model,
                    Optimizer optimizer,
                    ReplayBuffer replayBuffer,
                    float gamma,
                    float entropyCoeff) {
        // 1. super() 必须是第一条语句
        super(model, optimizer, replayBuffer);
        // 2. 初始化自定义超参数
        this.gamma = gamma;
        this.entropyCoeff = entropyCoeff;
    }

    // ===================== 简化构造函数（修正核心）=====================
    /**
     * 简化构造函数：快速创建 A2CAgent（默认超参数）
     * 修正：通过「静态内部类/工具方法」提前构建组件，确保 super() 是第一条语句
     * @param stateDim 状态维度
     * @param actionDim 动作维度
     * @param actionSpaceType 动作空间类型（DISCRETE/CONTINUOUS）
     */
    /**
     * 简化构造：同一 model 实例绑定 optimizer（不再 close 临时模型）。
     */
    public A2CAgent(long stateDim, long actionDim, AbstractActorCritic.ActionSpaceType actionSpaceType) {
        this(build(stateDim, actionDim, actionSpaceType, 3e-4f), 0.99f, 0.01f);
    }

    /** 工厂：可自定义 gamma / entropy / lr。 */
    public static A2CAgent create(long stateDim, long actionDim,
                                  AbstractActorCritic.ActionSpaceType actionSpaceType,
                                  float gamma, float entropyCoeff, float lr) {
        return new A2CAgent(build(stateDim, actionDim, actionSpaceType, lr), gamma, entropyCoeff);
    }

    public static A2CAgent create(long stateDim, long actionDim,
                                  AbstractActorCritic.ActionSpaceType actionSpaceType) {
        return create(stateDim, actionDim, actionSpaceType, 0.99f, 0.01f, 3e-4f);
    }

    /** Private bridge so super() can receive a single pre-built (model, optimizer) pair. */
    private A2CAgent(Object[] built, float gamma, float entropyCoeff) {
        super((AbstractActorCritic) built[0], (Optimizer) built[1], new ReplayBuffer());
        this.gamma = gamma;
        this.entropyCoeff = entropyCoeff;
    }

    private static Object[] build(long stateDim, long actionDim,
                                  AbstractActorCritic.ActionSpaceType actionSpaceType, float lr) {
        AbstractActorCritic model = createActorCriticModel(stateDim, actionDim, actionSpaceType);
        AdamOptions optOptions = new AdamOptions();
        optOptions.lr().put(lr);
        Optimizer optimizer = new Adam(model.parameters(), optOptions);
        return new Object[]{model, optimizer};
    }

    private static AbstractActorCritic createActorCriticModel(long stateDim, long actionDim,
                                                              AbstractActorCritic.ActionSpaceType actionSpaceType) {
        if (actionSpaceType == AbstractActorCritic.ActionSpaceType.CONTINUOUS) {
            return new ActorCritic(stateDim, actionDim);
        }
        return new CartPoleActorCritic(stateDim, actionDim);
    }

    // ===================== 实现抽象方法（核心要求）=====================
    @Override
    public Tensor trainStep() {
        ReplayBuffer buffer = super.getReplayBuffer();
        if (buffer.size() == 0) {
            throw new IllegalStateException("经验缓冲区为空，无法执行训练！");
        }

        Tensor states = buffer.getStates();
        // Discrete actions stack as [T,1] — squeeze to [T] before Categorical.log_prob
        Tensor actions = flattenTrailing(buffer.getActions());
        Tensor advantages = flattenTrailing(buffer.getAdvantages());
        Tensor returns = flattenTrailing(buffer.getReturns());

        AbstractActorCritic model = super.getModel();
        Distribution dist = model.getDistribution(states);

        Tensor logProbs = sumActionLogProb(dist.log_prob(actions));
        logProbs = flattenTrailing(logProbs);
        Tensor actorLoss = logProbs.mul(advantages.detach()).mean().neg();

        Tensor values = flattenTrailing(model.getValue(states));
        Tensor criticLoss = mse_loss(values, returns);

        Tensor entropy = dist.entropy().mean();
        Tensor entropyBonus = entropy.mul(new Scalar(this.entropyCoeff));

        Tensor totalLoss = actorLoss.add(criticLoss).sub(entropyBonus);

        Optimizer optimizer = super.optimizer;
        optimizer.zero_grad();
        totalLoss.backward();
        torch.clip_grad_norm_(model.parameters(), 0.5);
        optimizer.step();

        return totalLoss.detach();
    }

    /** Collapse trailing size-1 dims; keep rank ≥1. Mirrors PPOAgent.flattenBatch. */
    private static Tensor flattenTrailing(Tensor t) {
        Tensor x = t;
        while (x.dim() > 1 && x.size(x.dim() - 1) == 1) {
            x = x.squeeze(x.dim() - 1);
        }
        if (x.dim() == 0) {
            x = x.reshape(1);
        }
        return x;
    }

    @Override
    public Tensor[] sample(Tensor state) {
        AbstractActorCritic model = super.getModel();
        model.train(true);

        Distribution dist = model.getDistribution(state);
        Tensor action = dist.sample();
        Tensor logProb = sumActionLogProb(dist.log_prob(action));
        Tensor value = model.getValue(state);

        return new Tensor[]{
                action.detach().clone(),
                logProb.detach().clone(),
                value.detach().clone()
        };
    }

    // ===================== 扩展方法（保留 A2C 特色）=====================
    /**
     * 手动更新方法（兼容原有代码）
     */
    public void update(Tensor states, Tensor actions, Tensor advantages, Tensor returns) {
        if (states == null || actions == null || advantages == null || returns == null) {
            throw new IllegalArgumentException("输入张量不能为空！");
        }

        AbstractActorCritic model = super.getModel();
        Distribution dist = model.getDistribution(states);

        Tensor acts = flattenTrailing(actions);
        Tensor adv = flattenTrailing(advantages);
        Tensor ret = flattenTrailing(returns);
        Tensor logProbs = flattenTrailing(sumActionLogProb(dist.log_prob(acts)));
        Tensor actorLoss = logProbs.mul(adv.detach()).mean().neg();
        Tensor values = flattenTrailing(model.getValue(states));
        Tensor criticLoss = mse_loss(values, ret);
        Tensor entropy = dist.entropy().mean();
        Tensor totalLoss = actorLoss.add(criticLoss).sub(entropy.mul(new Scalar(this.entropyCoeff)));

        super.optimizer.zero_grad();
        totalLoss.backward();
        torch.clip_grad_norm_(model.parameters(), 0.5);
        super.optimizer.step();
    }

    /**
     * 计算折扣回报（A2C 基础版）
     */
    public Tensor computeDiscountedReturns(Tensor rewards, Tensor dones, Tensor lastValue) {
        long T = rewards.size(0);
        Tensor returns = torch.zeros_like(rewards);
        float runningReturn = lastValue.item().toFloat();

        for (long t = T - 1; t >= 0; t--) {
            runningReturn = rewards.select(0, t).item().toFloat()
                    + this.gamma * runningReturn * (1 - dones.select(0, t).item().toFloat());
            returns.select(0, t).fill_(new Scalar(runningReturn));
        }

        return returns;
    }

    // ===================== Getter/Setter =====================
    public float getEntropyCoeff() {
        return entropyCoeff;
    }

    public float getGamma() {
        return gamma;
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
