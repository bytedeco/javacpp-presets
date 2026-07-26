package org.bytedeco.pytorch.rl.agent;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.LMActorCritic;

/**
 * RL Agent 抽象父类
 * 统一所有训练器的核心行为（训练、采样、资源管理）
 */
public abstract class AbstractRLAgent {
    protected final AbstractActorCritic model; // 策略-价值模型
    protected final Optimizer optimizer;       // 优化器
    protected final ReplayBuffer replayBuffer; // 经验缓冲区

    /**
     * 构造函数
     * @param model 策略-价值模型
     * @param optimizer 优化器
     * @param replayBuffer 经验缓冲区
     */
    public AbstractRLAgent(AbstractActorCritic model, Optimizer optimizer, ReplayBuffer replayBuffer) {
        this.model = model;
        this.optimizer = optimizer;
        this.replayBuffer = replayBuffer;
    }

    /**
     * 单步训练（核心方法）
     * @return 训练损失
     */
    public abstract Tensor trainStep();

    protected void freezeModel(AbstractActorCritic model) {
        var paramsVector = model.parameters();
        var begin = paramsVector.begin();
        var end = paramsVector.end();
        while(!begin.equals(end)) {
            var param = begin.get();
            param.requires_grad_(false);
            begin.increment();
        }
        paramsVector.close();
    }
    /**
     * 采样动作（收集经验）
     * @param state 状态张量
     * @return 动作 + 对数概率 + 价值（按需返回）
     */
    public abstract Tensor[] sample(Tensor state);

    /**
     * 清空缓冲区
     */
    public void clearBuffer() {
        if (replayBuffer != null) {
            replayBuffer.clear();
        }
    }

    /**
     * 释放所有资源（统一接口）
     */
    public void close() {
        if (model != null) model.close();
        if (optimizer != null) optimizer.close();
        if (replayBuffer != null) replayBuffer.clear();
    }

    // 通用Getter
    public AbstractActorCritic getModel() { return model; }
    public ReplayBuffer getReplayBuffer() { return replayBuffer; }
}