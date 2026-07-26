package org.bytedeco.pytorch.rl.critic;
//package org.bytedeco.pytorch.rl.core;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Distribution;

/**
 * Actor-Critic 网络抽象父类
 * 统一策略网络（Actor）和价值网络（Critic）的核心接口
 */
public abstract class AbstractActorCritic extends Module implements DistributionProvider {
    // 动作空间类型：DISCRETE/CONTINUOUS
    public enum ActionSpaceType {
        DISCRETE, CONTINUOUS
    }

    protected final long stateDim;      // 状态/输入维度
    protected final long actionDim;     // 动作维度
    protected final ActionSpaceType actionSpaceType; // 动作空间类型

    /**
     * 构造函数
     * @param stateDim 状态维度
     * @param actionDim 动作维度
     * @param actionSpaceType 动作空间类型
     */
    public AbstractActorCritic(long stateDim, long actionDim, ActionSpaceType actionSpaceType) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.actionSpaceType = actionSpaceType;
    }

    /**
     * 获取状态价值 V(s)（核心Critic方法）
     * @param state 状态张量
     * @return 价值张量
     */
    public abstract Tensor getValue(Tensor state);

    /**
     * 获取动作分布（核心Actor方法，实现DistributionProvider接口）
     */
    @Override
    public abstract Distribution getDistribution(Tensor state);

    /**
     * 释放资源（统一接口）
     */
    @Override
    public abstract void close();

    // 通用Getter
    public long getStateDim() { return stateDim; }
    public long getActionDim() { return actionDim; }
    public ActionSpaceType getActionSpaceType() { return actionSpaceType; }
}