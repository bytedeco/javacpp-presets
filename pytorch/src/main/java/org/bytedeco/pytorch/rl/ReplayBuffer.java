package org.bytedeco.pytorch.rl;
import org.bytedeco.pytorch.data.transforms.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 通用强化学习经验缓冲区
 * 支持：
 * 1. 基础PPO所需：状态、动作、对数概率、奖励、价值、结束标志
 * 2. GRPO所需：环境奖励、引导奖励
 * 3. LMPPO所需：长度掩码、序列级奖励、token级数据
 * 4. 通用GAE计算、内存安全管理、多维度张量兼容
 */
public class ReplayBuffer {
    // ===================== 基础核心字段（所有算法通用）=====================
    private final List<Tensor> states = new ArrayList<>();          // 状态 [batch, ...]
    private final List<Tensor> actions = new ArrayList<>();         // 动作 [batch, ...]
    private final List<Tensor> logProbs = new ArrayList<>();        // 对数概率（旧策略）
    private final List<Tensor> rewards = new ArrayList<>();         // 基础奖励（兼容旧代码）
    private final List<Tensor> values = new ArrayList<>();          // 价值预测
    private final List<Tensor> dones = new ArrayList<>();           // 结束标志（0/1）

    // ===================== GRPO扩展字段 =====================
    private final List<Tensor> envRewards = new ArrayList<>();      // 环境奖励
    private final List<Tensor> guideRewards = new ArrayList<>();    // 引导奖励

    // ===================== LMPPO扩展字段 =====================
    private final List<Tensor> masks = new ArrayList<>();           // 长度掩码（1=有效token，0=填充）
    private final List<Tensor> seqRewards = new ArrayList<>();      // 序列级奖励（[batch]）

    // ===================== 预计算 advantages / returns 列表（push / push2）=====================
    private final List<Tensor> advantages = new ArrayList<>();
    private final List<Tensor> returns = new ArrayList<>();

    // ===================== 预计算张量（训练时使用）=====================
    private Tensor t_states, t_actions, t_logProbs, t_returns, t_advantages;
    private Tensor t_envRewards, t_guideRewards, t_masks, t_seqRewards, t_dones, t_values;

    // ===================== 基础API（兼容原有逻辑）=====================
    public List<Tensor> getStateList() { return states; }
    public int size() { return states.size(); }

    // ===================== 新增：获取各类扩展数据的API（核心改造）=====================
    // 基础PPO/GRPO/LMPPO通用
    public Tensor getOldLogProbs() { return this.t_logProbs != null ? this.t_logProbs : stackList(logProbs); }
    public Tensor getDones() { return this.t_dones != null ? this.t_dones : stackList(dones); }
    public Tensor getValues() { return this.t_values != null ? this.t_values : stackList(values); }

    // GRPO专用
    public Tensor getEnvRewards() { return this.t_envRewards != null ? this.t_envRewards : stackList(envRewards); }
    public Tensor getGuideRewards() { return this.t_guideRewards != null ? this.t_guideRewards : stackList(guideRewards); }

    // LMPPO专用
    public Tensor getMasks() { return this.t_masks != null ? this.t_masks : stackList(masks); }
    public Tensor getRewards() { return this.t_seqRewards != null ? this.t_seqRewards : stackList(seqRewards); }

    // ===================== 数据存入方法（重载，适配不同算法）=====================
    /**
     * 基础存入：适配PPO（状态+动作+对数概率+奖励+结束标志+价值）
     */
    public void add(Tensor s, Tensor a, Tensor r, Tensor lp, Tensor v, Tensor done) {
        addTensor(states, s);
        addTensor(actions, a);
        addTensor(rewards, r);
        addTensor(logProbs, lp);
        addTensor(values, v);
        addTensor(dones, done);
    }

    /**
     * GRPO专用存入：环境奖励+引导奖励
     */
    public void addGRPO(Tensor s, Tensor a, Tensor lp, Tensor envR, Tensor guideR, Tensor done, Tensor v) {
        addTensor(states, s);
        addTensor(actions, a);
        addTensor(logProbs, lp);
        addTensor(envRewards, envR);
        addTensor(guideRewards, guideR);
        addTensor(dones, done);
        addTensor(values, v);
    }

    /**
     * LMPPO专用存入：序列奖励+长度掩码+token级价值
     */
    public void addLMPPO(Tensor s, Tensor a, Tensor lp, Tensor seqR, Tensor mask, Tensor v) {
        addTensor(states, s);
        addTensor(actions, a);
        addTensor(logProbs, lp);
        addTensor(seqRewards, seqR);
        addTensor(masks, mask);
        addTensor(values, v);
    }

    /**
     * 兼容原有push方法：适配旧代码逻辑
     */
    public void push(Tensor s, Tensor a, Tensor lp, Tensor adv, Tensor ret) {
        // Normalize to rank ≥1 so stack() yields [T, ...] not ragged ranks.
        // Discrete actions / scalar log-probs / adv / ret are forced to shape [1].
        Tensor s1 = atLeast1D(s);
        Tensor a1 = squeezeTrailingOnes(atLeast1D(a));
        Tensor lp1 = squeezeTrailingOnes(atLeast1D(lp));
        Tensor adv1 = squeezeTrailingOnes(atLeast1D(adv));
        Tensor ret1 = squeezeTrailingOnes(atLeast1D(ret));

        addTensor(states, s1);
        addTensor(actions, a1);
        addTensor(logProbs, lp1);
        addTensor(rewards, adv1);       // 兼容旧代码：rewards暂存advantages
        addTensor(advantages, adv1);
        addTensor(returns, ret1);
        addTensor(values, ret1);        // 兼容旧代码：values暂存returns
    }

    private static Tensor atLeast1D(Tensor t) {
        return t.dim() == 0 ? t.reshape(1) : t;
    }

    /** Collapse trailing size-1 dims but keep at least rank 1. */
    private static Tensor squeezeTrailingOnes(Tensor t) {
        Tensor x = t;
        while (x.dim() > 1 && x.size(x.dim() - 1) == 1) {
            x = x.squeeze(x.dim() - 1);
        }
        if (x.dim() == 0) {
            x = x.reshape(1);
        }
        return x;
    }

    /**
     * 兼容原有push2方法
     */
    public void push2(Tensor s, Tensor a, Tensor lp, Tensor adv, Tensor ret) {
        addTensor(states, s);
        addTensor(actions, a);
        addTensor(logProbs, lp);
        addTensor(advantages, adv);
        addTensor(returns, ret);
    }

    // ===================== 核心：GAE构建方法（重构，支持多类型奖励）=====================
    /**
     * 通用GAE构建：支持基础奖励/环境奖励/引导奖励，兼容PPO/GRPO
     * @param lastValue 最后一步的价值预测
     * @param gamma 折扣因子
     * @param gaeLambda GAE系数
     * @param useGuideReward 是否使用GRPO的引导奖励（true=使用融合奖励，false=使用基础奖励）
     */
    public void build(float lastValue, float gamma, float gaeLambda, boolean useGuideReward) {
        int size = states.size();
        if (size == 0) return;

        // 1. 堆叠基础张量（所有算法通用）
        this.t_states = stackList(states);
        this.t_actions = stackList(actions);
        this.t_logProbs = stackList(logProbs);
        this.t_values = stackList(values);
        this.t_dones = stackList(dones);

        // 2. 选择奖励类型（PPO=基础奖励，GRPO=融合奖励）
        Tensor targetRewards;
        if (useGuideReward && !envRewards.isEmpty() && !guideRewards.isEmpty()) {
            // GRPO：融合环境奖励+引导奖励（默认权重0.5）
            this.t_envRewards = stackList(envRewards);
            this.t_guideRewards = stackList(guideRewards);
            targetRewards = this.t_envRewards.mul(new Scalar(0.5f)).add(this.t_guideRewards.mul(new Scalar(0.5f)));
        } else {
            // PPO：使用基础奖励
            targetRewards = stackList(rewards);
        }

        // 3. GAE核心计算（从后往前迭代）
        float[] advantages = new float[size];
        float[] returns = new float[size];
        float lastGae = 0;
        float nextValue = lastValue;

        for (int t = size - 1; t >= 0; t--) {
            // 提取标量值（兼容0D/1D张量）
            float r = getScalarValue(targetRewards, t);
            float v = getScalarValue(this.t_values, t);
            float done = getScalarValue(this.t_dones, t);
            float mask = 1.0f - done; // done=1时mask=0，终止GAE累积

            // TD误差 = 奖励 + γ*下一个价值*mask - 当前价值
            float delta = r + gamma * nextValue * mask - v;
            // GAE优势 = delta + γ*λ*mask*上一步GAE
            advantages[t] = lastGae = delta + gamma * gaeLambda * mask * lastGae;
            // 回报 = 优势 + 当前价值
            returns[t] = advantages[t] + v;

            nextValue = v;
        }

        // 4. 转换为Tensor并存储
        this.t_advantages = tensor(advantages);
        this.t_returns = tensor(returns);

        // 5. LMPPO额外处理：堆叠掩码和序列奖励
        if (!masks.isEmpty()) this.t_masks = stackList(masks);
        if (!seqRewards.isEmpty()) this.t_seqRewards = stackList(seqRewards);
    }

    /**
     * 简化GAE构建：适配原有PPO逻辑（不使用引导奖励）
     */
    public void build(float lastValue, float gamma, float gaeLambda) {
        build(lastValue, gamma, gaeLambda, false);
    }

    // ===================== 辅助方法（工具类）=====================
    /**
     * 安全添加张量：detach+clone防止梯度图污染和内存泄漏
     */
    private void addTensor(List<Tensor> list, Tensor t) {
        if (t == null || t.isNull()) {
            throw new IllegalArgumentException("传入的Tensor不能为空或已释放！");
        }
        // 移除梯度图 + 独立内存拷贝
        list.add(t.detach().clone());
    }

    /**
     * 堆叠List<Tensor>为批量Tensor（兼容空列表）
     */
    private Tensor stackList(List<Tensor> list) {
        if (list.isEmpty()) return null;
        try (TensorVector vector = new TensorVector()) {
            for (Tensor t : list) {
                // MUST use push_back — put() overwrites index 0 and collapses the batch to size 1
                // (which then makes advantage.std() / MSE shapes explode into NaN).
                if (t != null && !t.isNull()) vector.push_back(t);
            }
            if (vector.empty()) return null;
            return stack(vector);
        }
    }

    /**
     * 从张量中提取标量值（兼容任意维度）
     */
    private float getScalarValue(Tensor tensor, int index) {
        if (tensor == null || tensor.isNull()) return 0.0f;
        // 处理1D张量：直接取index位置；处理高维张量：先squeeze再取
        Tensor scalarTensor = tensor.dim() > 1 ? tensor.squeeze().select(0, index) : tensor.select(0, index);
        return scalarTensor.item().toFloat();
    }

    // ===================== 原有方法兼容 =====================
    public Tensor getStates() { return this.t_states != null ? this.t_states : stackList(states); }
    public Tensor getActions() { return this.t_actions != null ? this.t_actions : stackList(actions); }
    public Tensor getLogProbs() { return this.t_logProbs != null ? this.t_logProbs : stackList(logProbs); }
    public Tensor getReturns() { return this.t_returns != null ? this.t_returns : stackList(returns); }
    public Tensor getAdvantages() { return this.t_advantages != null ? this.t_advantages : stackList(advantages); }

    public Tensor getAllStates() {
        if (states.isEmpty()) return null;
        try (TensorVector vector = new TensorVector()) {
            for (Tensor t : states) {
                if (t == null || t.isNull()) {
                    throw new RuntimeException("检测到空Tensor，内存可能已提前释放！");
                }
                vector.push_back(t);
            }
            return cat(vector, 0);
        }
    }

    public Tensor[] getAll() {
        if (states.isEmpty()) return null;
        try (TensorVector vS = new TensorVector();
             TensorVector vA = new TensorVector();
             TensorVector vL = new TensorVector();
             TensorVector vAdv = new TensorVector();
             TensorVector vRet = new TensorVector()) {

            for (int i = 0; i < states.size(); i++) {
                if (states.get(i).isNull()) continue;
                vS.push_back(states.get(i));
                vA.push_back(actions.get(i));
                vL.push_back(logProbs.get(i));
                vAdv.push_back(advantages.get(i));
                vRet.push_back(returns.get(i));
            }

            return new Tensor[]{
                    cat(vS, 0).detach().clone(),
                    cat(vA, 0).detach().clone(),
                    cat(vL, 0).detach().clone(),
                    cat(vAdv, 0).detach().clone(),
                    cat(vRet, 0).detach().clone()
            };
        }
    }

    // ===================== 内存管理（关键优化）=====================
    /**
     * 清空缓冲区并释放所有Tensor内存（防止Native内存泄漏）
     */
    public void clear() {
        // 释放基础字段
        closeList(states);
        closeList(actions);
        closeList(logProbs);
        closeList(rewards);
        closeList(values);
        closeList(dones);

        // 释放扩展字段
        closeList(envRewards);
        closeList(guideRewards);
        closeList(masks);
        closeList(seqRewards);
        closeList(advantages);
        closeList(returns);

        // 释放预计算张量
        closeTensor(t_states);
        closeTensor(t_actions);
        closeTensor(t_logProbs);
        closeTensor(t_returns);
        closeTensor(t_advantages);
        closeTensor(t_envRewards);
        closeTensor(t_guideRewards);
        closeTensor(t_masks);
        closeTensor(t_seqRewards);
        closeTensor(t_dones);
        closeTensor(t_values);

        // 清空列表
        states.clear();
        actions.clear();
        logProbs.clear();
        rewards.clear();
        values.clear();
        dones.clear();
        envRewards.clear();
        guideRewards.clear();
        masks.clear();
        seqRewards.clear();
        advantages.clear();
        returns.clear();
        t_states = t_actions = t_logProbs = t_returns = t_advantages = null;
        t_envRewards = t_guideRewards = t_masks = t_seqRewards = t_dones = t_values = null;
    }

    /**
     * 批量释放List中的Tensor
     */
    private void closeList(List<Tensor> list) {
        for (Tensor t : list) {
            if (t != null && !t.isNull()) {
                t.close();
            }
        }
    }

    /**
     * 释放单个Tensor（兼容null）
     */
    private void closeTensor(Tensor t) {
        if (t != null && !t.isNull()) {
            t.close();
        }
    }
}
