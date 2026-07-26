package org.bytedeco.pytorch.rl.critic;

//package org.bytedeco.pytorch.rl.core;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Distribution;

/**
 * 策略分布提供器接口
 * 所有生成动作分布的网络都需实现此接口
 */
public interface DistributionProvider {
    /**
     * 获取动作分布（核心方法）
     * @param state 状态/隐藏层张量
     * @return 概率分布（Normal/Categorical）
     */
    Distribution getDistribution(Tensor state);
}