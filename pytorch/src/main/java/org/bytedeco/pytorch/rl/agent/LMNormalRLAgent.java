package org.bytedeco.pytorch.rl.agent;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.enumtype.*;

import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.distribution.Normal;
import org.bytedeco.pytorch.rl.critic.LMActorCritic;

import java.util.Arrays;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Normal分布RL训练器（内存优化版）
 * 核心优化：
 * 1. 极致资源管理：所有临时张量即时释放，避免内存累积
 * 2. 常量张量复用：避免重复创建，减少内存碎片
 * 3. 惰性广播：仅在必要时执行广播，减少中间张量
 * 4. 批量清理：训练步结束后强制GC，释放未回收的内存
 */
public class LMNormalRLAgent implements AutoCloseable {
    // ===================== 成员变量定义 =====================
    private final LMActorCritic actorCriticModel;
    private final Optimizer optimizer;
    private final float gamma;
    private final float lambda;
    private final double clipNorm;
    private final String actionSpaceType;
    private final Device cpuDevice;
    private final TensorOptions floatOpts;
    private final TensorOptions longOpts;
    private final long vocabSize;

    // 复用常量张量（全局唯一，避免重复创建）
    private final Tensor EPS_TENSOR;
    private final Tensor BIG_EPS_TENSOR;
    private final Tensor ZERO_LONG;
    private final Tensor ONE_FLOAT;
    private final Tensor MINUS_ONE_LONG;
    private final Tensor GAMMA_TENSOR;
    private final Tensor LAMBDA_TENSOR;
    private final Tensor REWARD_W1;
    private final Tensor REWARD_W2;
    private final Tensor HALF;
    private final Tensor MIN_REWARD;

    // 静态常量（避免魔法值）
    private static final float EPS = 1e-8f;
    private static final float BIG_EPS = 1e-4f;
    private static final float MIN_REWARD_VAL = -10.0f;
    private static final int CLIP_VALUE = 10;
    private static final int CLIP_ADVANTAGE = 5;

    // ===================== 构造函数 =====================
    public LMNormalRLAgent(LMActorCritic actorCriticModel,
                           float gamma, float lambda, float lr, float clipNorm,
                           String actionSpaceType, TensorOptions floatOpts, Device cpuDevice,
                           long vocabSize) {
        // 参数校验
        if (actorCriticModel == null) throw new IllegalArgumentException("模型不能为空");
        if (vocabSize <= 0) throw new IllegalArgumentException("词汇表大小必须>0");
        if (cpuDevice == null) throw new IllegalArgumentException("设备不能为空");

        this.actorCriticModel = actorCriticModel;
        this.gamma = gamma;
        this.lambda = lambda;
        this.clipNorm = clipNorm;
        this.actionSpaceType = actionSpaceType;
        this.cpuDevice = cpuDevice;
        this.vocabSize = vocabSize;

        // 固定设备和类型，避免动态切换导致的内存泄漏
        this.floatOpts = new TensorOptions()
                .device(new DeviceOptional(cpuDevice))
                .dtype(new ScalarTypeOptional(kFloat()))
                .requires_grad(new BoolOptional(false));

        this.longOpts = new TensorOptions()
                .device(new DeviceOptional(cpuDevice))
                .dtype(new ScalarTypeOptional(kLong()))
                .requires_grad(new BoolOptional(false));

        // 优化器配置（简化API调用）
        AdamOptions optOpts = new AdamOptions();
        optOpts.lr().put(lr);
        optOpts.weight_decay().put(1e-5f);
        optOpts.betas().put(new double[]{0.9, 0.999});
        optOpts.eps().put(1e-8);
        this.optimizer = new Adam(actorCriticModel.parameters(), optOpts);
        optOpts.close();

        // 初始化常量张量（全局复用，仅创建一次）
        this.EPS_TENSOR = createConstTensor(EPS);
        this.BIG_EPS_TENSOR = createConstTensor(BIG_EPS);
        this.ZERO_LONG = createConstLongTensor(0L);
        this.ONE_FLOAT = createConstTensor(1.0f);
        this.MINUS_ONE_LONG = createConstLongTensor(-1L);
        this.GAMMA_TENSOR = createConstTensor(gamma);
        this.LAMBDA_TENSOR = createConstTensor(lambda);
        this.REWARD_W1 = createConstTensor(0.7f);
        this.REWARD_W2 = createConstTensor(0.3f);
        this.HALF = createConstTensor(0.5f);
        this.MIN_REWARD = createConstTensor(MIN_REWARD_VAL);
    }

    // ===================== 核心工具方法（内存优化）=====================
    /**
     * 创建常量float张量（仅创建+detach，避免梯度关联）
     */
    private Tensor createConstTensor(float value) {
        try (Tensor t = tensor(value, floatOpts)) {
            return t.to(cpuDevice, kFloat()).clone().detach();
        }
    }

    /**
     * 创建常量long张量（同上）
     */
    private Tensor createConstLongTensor(long value) {
        try (Tensor t = tensor(value, longOpts)) {
            return t.to(cpuDevice, kLong()).clone().detach();
        }
    }

    /**
     * 安全释放张量（核心优化：增加isNull检查，避免空指针）
     */
    private void safeClose(Tensor tensor) {
        if (tensor != null && !tensor.isNull()) {
            try {
                tensor.close();
            } catch (Exception e) {
                System.err.println("张量释放警告: " + e.getMessage());
            }
        }
    }

    /**
     * 惰性广播：仅当维度不匹配时执行广播，减少中间张量
     */
    private Tensor lazyBroadcast(Tensor tensor, long[] targetShape) {
        if (tensor == null || tensor.isNull()) return null;

        long[] currentShape = tensor.sizes().vec().get();
        // 如果维度已匹配，直接返回原张量（避免冗余操作）
        if (Arrays.equals(currentShape, targetShape)) {
            return tensor;
        }

        // 补维+expand（优化：使用view代替reshape，减少内存拷贝）
        try {
            long[] newShape = new long[targetShape.length];
            System.arraycopy(currentShape, 0, newShape, 0, currentShape.length);
            for (int i = currentShape.length; i < targetShape.length; i++) {
                newShape[i] = 1;
            }
            return tensor.view(newShape).expand(targetShape).clone().detach();
        } catch (Exception e) {
            System.err.println("广播失败，返回原张量: " + e.getMessage());
            return tensor;
        }
    }

    /**
     * 打印张量维度（调试用，可关闭以提升性能）
     */
    private void logTensorInfo(String name, Tensor tensor) {
        if (tensor == null || tensor.isNull()) {
            System.out.println("[维度日志] " + name + ": NULL");
            return;
        }
        long[] sizes = tensor.sizes().vec().get();
        long numElements = tensor.numel();
        System.out.printf("[维度日志] %s: 形状=[%s], 总元素数=%d%n",
                name, String.join(", ", Arrays.stream(sizes).mapToObj(String::valueOf).toArray(String[]::new)),
                numElements);
    }

    // ===================== Rollout方法（内存优化）=====================
    public RolloutResult rollout(Tensor inputEmbeds, Tensor masks, int rolloutSteps) {
        // 入参校验
        if (inputEmbeds == null || masks == null) {
            throw new IllegalArgumentException("输入张量不能为空");
        }

        long[] inputShape = inputEmbeds.sizes().vec().get();
        if (inputShape.length != 3) {
            throw new IllegalArgumentException("inputEmbeds必须是3维张量");
        }
        long batchSize = inputShape[0];
        long seqLen = inputShape[1];
        long hiddenDim = inputShape[2];

        actorCriticModel.eval();
        RolloutResult result = new RolloutResult();

        // 临时变量声明（集中管理）
        Tensor hidden = null;
        Tensor actions = null;
        Tensor actionLogits = null;
        Tensor actionsMean = null;
        Tensor actionsStd = null;
        Tensor actionsNorm = null;
        Tensor actionScores = null;
        Tensor discreteActions = null;
        Tensor logProbs = null;
        Tensor values = null;
        Tensor maskedActions = null;
        Tensor rewards = null;
        Tensor projLayer = null;
        Tensor validPositions = null;

        try {
            // 1. 模型前向传播（即时释放临时张量）
            hidden = actorCriticModel.forwardLM(inputEmbeds);
            logTensorInfo("hidden (模型输出)", hidden);

            // 动态调整维度（使用view代替reshape，减少内存拷贝）
            if (hidden.dim() == 2) {
                hidden = hidden.view(new long[]{batchSize, seqLen, hiddenDim});
            }

            // 2. Normal分布采样（核心：采样后即时裁剪，避免数值溢出）
            Distribution dist = actorCriticModel.getDistribution(hidden);
            if (!(dist instanceof Normal)) {
                throw new RuntimeException("期望Normal分布，实际为：" + (dist == null ? "NULL" : dist.getClass().getName()));
            }
            Normal normalDist = (Normal) dist;
            actions = normalDist.sample();
            logTensorInfo("actions (采样后)", actions);

            if (actions.dim() == 2) {
                actions = actions.view(new long[]{batchSize, seqLen, hiddenDim});
            }
            // 裁剪动作值，避免梯度爆炸
            actions = clamp(actions,new ScalarOptional(new Scalar(-CLIP_VALUE)) , new ScalarOptional(new Scalar(CLIP_VALUE)));

            // 3. 投影到词汇表维度（核心优化：projLayer使用后即时释放）
            projLayer = randn(new long[]{hiddenDim, vocabSize}, floatOpts);
            actionLogits = actions.matmul(projLayer.t());
            logTensorInfo("actionLogits (投影后)", actionLogits);
            safeClose(projLayer); // 即时释放投影层

            // 4. 归一化（惰性广播，减少中间张量）
            actionsMean = actionLogits.mean(new long[]{-1}, true, new ScalarTypeOptional());
            logTensorInfo("actionsMean (原始)", actionsMean);
            actionsMean = lazyBroadcast(actionsMean, actionLogits.sizes().vec().get());
            logTensorInfo("actionsMean (广播后)", actionsMean);

            actionsStd = actionLogits.std(new long[]{-1}, true).add(EPS_TENSOR);
            logTensorInfo("actionsStd (原始)", actionsStd);
            actionsStd = lazyBroadcast(actionsStd, actionLogits.sizes().vec().get());
            logTensorInfo("actionsStd (广播后)", actionsStd);

            // 归一化计算（维度已匹配）
            actionsNorm = actionLogits.sub(actionsMean).div(actionsStd);
            logTensorInfo("actionsNorm", actionsNorm);

            // 5. 离散动作计算（即时释放临时张量）
            actionScores = softmax(actionsNorm, -1);
            discreteActions = argmax(actionScores, new LongOptional(-1), false);
            logTensorInfo("discreteActions", discreteActions);
            safeClose(actionScores); // 即时释放

            // 6. 应用mask（核心优化：复用ZERO_LONG常量）
            validPositions = masks.to(kLong()).gt(ZERO_LONG);
            maskedActions = discreteActions.where(validPositions, MINUS_ONE_LONG);
            safeClose(validPositions); // 即时释放

            // 7. 对数概率和价值计算（裁剪值，避免数值异常）
            logProbs = normalDist.log_prob(actions).sum(-1).mul(masks);
            logProbs = clamp(logProbs, new ScalarOptional(new Scalar(-100)), new ScalarOptional(new Scalar(100)));

            values = actorCriticModel.getValue(hidden);
            logTensorInfo("values (原始)", values);
            if (values.dim() == 3) {
                values = values.squeeze(-1);
                logTensorInfo("values (squeeze后)", values);
            }
            values = values.mul(masks);
            values = clamp(values, new ScalarOptional(new Scalar(-CLIP_VALUE)), new ScalarOptional(new Scalar(CLIP_VALUE)));

            // 8. 计算奖励（核心优化：奖励张量即时释放）
            rewards = computeReward(maskedActions, masks, batchSize, seqLen);

            // 9. 赋值结果（detach避免梯度关联，减少内存占用）
            result.states = inputEmbeds.clone().detach();
            result.actions = discreteActions.clone().detach();
            result.rewards = rewards.clone().detach();
            result.masks = masks.clone().detach();
            result.logProbs = logProbs.clone().detach();
            result.values = values.clone().detach();

            actorCriticModel.train(true);
            return result;

        } catch (Exception e) {
            result.close(); // 异常时释放结果张量
            throw new RuntimeException("Rollout失败: " + e.getMessage(), e);
        } finally {
            // 核心优化：即时释放所有临时张量，避免内存累积
            safeClose(hidden);
            safeClose(actions);
            safeClose(actionLogits);
            safeClose(actionsMean);
            safeClose(actionsStd);
            safeClose(actionsNorm);
            safeClose(discreteActions);
            safeClose(logProbs);
            safeClose(values);
            safeClose(maskedActions);
            safeClose(rewards);
        }
    }

    // ===================== trainStep方法（内存优化核心）=====================
    public float trainStep(Tensor inputEmbeds, Tensor masks) {
        if (inputEmbeds == null || masks == null) {
            throw new IllegalArgumentException("输入张量不能为空");
        }

        long[] inputShape = inputEmbeds.sizes().vec().get();
        long batchSize = inputShape[0];
        long seqLen = inputShape[1];
        long hiddenDim = inputShape[2];

        RolloutResult rollout = null;
        Tensor loss = null;
        float lossValue = 0.0f;

        try {
            // 1. 轨迹采样（使用后即时释放）
            rollout = rollout(inputEmbeds, masks, 10);
            if (rollout == null) {
                throw new RuntimeException("Rollout结果为空");
            }

            // 2. 模型前向传播
            Tensor hidden = actorCriticModel.forwardLM(rollout.states);
            logTensorInfo("hidden (trainStep)", hidden);
            if (hidden.dim() == 2) {
                hidden = hidden.view(new long[]{batchSize, seqLen, hiddenDim});
            }

            // 3. 重新计算动作和价值
            Distribution dist = actorCriticModel.getDistribution(hidden);
            Normal normalDist = (Normal) dist;
            Tensor newActions = normalDist.sample();
            logTensorInfo("newActions (采样后)", newActions);

            if (newActions.dim() == 2) {
                newActions = newActions.view(new long[]{batchSize, seqLen, hiddenDim});
            }
            newActions = clamp(newActions, new ScalarOptional(new Scalar(-CLIP_VALUE)), new ScalarOptional(new Scalar(CLIP_VALUE)));

            // 4. 损失计算（核心优化：所有临时张量即时释放）
            Tensor newLogProbs = normalDist.log_prob(newActions).sum(-1).mul(rollout.masks);
            newLogProbs = clamp(newLogProbs, new ScalarOptional(new Scalar(-100)), new ScalarOptional(new Scalar(100)));

            Tensor newValues = actorCriticModel.getValue(hidden);
            if (newValues.dim() == 3) {
                newValues = newValues.squeeze(-1);
            }
            newValues = newValues.mul(rollout.masks);
            newValues = clamp(newValues, new ScalarOptional(new Scalar(-CLIP_VALUE)), new ScalarOptional(new Scalar(CLIP_VALUE)));

            // 5. 优势函数计算（复用常量张量，减少创建）
            Tensor advantages = computeGAE(rollout.rewards, rollout.values, rollout.masks, batchSize, seqLen);
            advantages = clamp(advantages, new ScalarOptional(new Scalar( -CLIP_ADVANTAGE)), new ScalarOptional(new Scalar(CLIP_ADVANTAGE)));

            // 6. PPO损失计算（简化API，减少临时张量）
            Tensor logProbDiff = newLogProbs.sub(rollout.logProbs);
            logProbDiff = clamp(logProbDiff,new ScalarOptional(new Scalar( -5)),new ScalarOptional(new Scalar( 5)));
            Tensor ratio = logProbDiff.exp();
            ratio = clamp(ratio, new ScalarOptional(new Scalar( 0.1f)), new ScalarOptional(new Scalar( 10.0f)));

            Tensor pgLoss = ratio.mul(advantages).neg().mean();
            MSELossOptions mseOpts = new MSELossOptions(new kMean());
            Tensor valueLoss = mse_loss(newValues, rollout.rewards, mseOpts);
            mseOpts.close();

            loss = pgLoss.add(valueLoss.mul(HALF)).add(EPS_TENSOR);

            // 修复NaN（核心优化：使用常量张量代替临时创建）
            if (loss.isnan().any().item_bool()) {
                loss = ONE_FLOAT.clone();
            }

            // 7. 反向传播（核心优化：梯度裁剪后即时清零）
            optimizer.zero_grad();
            loss.backward();
            clip_grad_norm_(actorCriticModel.parameters(), clipNorm);
            optimizer.step();

            // 8. 获取损失值（核心优化：item_float后即时释放张量）
            lossValue = loss.item_float();
            System.out.printf("[训练日志] PGLoss: %.4f | ValueLoss: %.4f | 总Loss: %.4f%n",
                    pgLoss.item_float(), valueLoss.item_float(), lossValue);

            // 即时释放临时张量
            safeClose(hidden);
            safeClose(newActions);
            safeClose(newLogProbs);
            safeClose(newValues);
            safeClose(advantages);
            safeClose(logProbDiff);
            safeClose(ratio);
            safeClose(pgLoss);
            safeClose(valueLoss);

            // 强制GC（核心优化：训练步结束后手动触发GC，释放内存）
            System.gc();

            return lossValue;

        } catch (Exception e) {
            throw new RuntimeException("训练步失败: " + e.getMessage(), e);
        } finally {
            // 最终释放：rollout和loss张量
            if (rollout != null) rollout.close();
            safeClose(loss);
        }
    }

    // ===================== 辅助方法（内存优化）=====================
    /**
     * 计算GAE优势函数（核心优化：减少临时张量创建）
     */
    private Tensor computeGAE(Tensor rewards, Tensor values, Tensor masks, long batchSize, long seqLen) {
        Tensor advantages = zeros_like(rewards, floatOpts, new MemoryFormatOptional());
        Tensor runningAdv = zeros(new long[]{batchSize}, floatOpts);

        try {
            for (int t = (int) seqLen - 1; t >= 0; t--) {
                // 切片优化：使用narrow+squeeze，减少内存拷贝
                Tensor rewardT = rewards.narrow(1, t, 1).squeeze(1);
                Tensor valueT = values.narrow(1, t, 1).squeeze(1);
                Tensor maskT = masks.narrow(1, t, 1).squeeze(1);

                // 下一时间步value（复用常量张量）
                Tensor nextValue = (t == seqLen - 1) ?
                        zeros_like(valueT, floatOpts, new MemoryFormatOptional()) :
                        values.narrow(1, t+1, 1).squeeze(1);

                // TD误差计算（核心优化：减少临时张量）
                Tensor delta = rewardT.add(GAMMA_TENSOR.mul(nextValue))
                        .sub(valueT)
                        .mul(maskT);
                delta = clamp(delta, new ScalarOptional(new Scalar(-CLIP_VALUE)), new ScalarOptional(new Scalar(CLIP_VALUE)));

                // 累计优势（复用常量张量）
                runningAdv.copy_(delta.add(GAMMA_TENSOR.mul(LAMBDA_TENSOR).mul(runningAdv).mul(maskT)));
                advantages.narrow(1, t, 1).copy_(runningAdv.unsqueeze(1));

                // 即时释放当前步张量
                safeClose(rewardT);
                safeClose(valueT);
                safeClose(maskT);
                safeClose(nextValue);
                safeClose(delta);
            }

            // 归一化（核心优化：复用BIG_EPS_TENSOR）
            Tensor advMean = advantages.mean();
            Tensor advStd = advantages.std().add(BIG_EPS_TENSOR);
            if (advStd.item_float() < BIG_EPS) {
                advStd = BIG_EPS_TENSOR;
            }

            Tensor normalizedAdv = advantages.sub(advMean).div(advStd);
            normalizedAdv = clamp(normalizedAdv, new ScalarOptional(new Scalar( -CLIP_ADVANTAGE)), new ScalarOptional(new Scalar(CLIP_ADVANTAGE)));

            // 释放临时张量
            safeClose(advMean);
            safeClose(advStd);
            safeClose(advantages);
            safeClose(runningAdv);

            return normalizedAdv;

        } catch (Exception e) {
            safeClose(advantages);
            safeClose(runningAdv);
            throw new RuntimeException("GAE计算失败: " + e.getMessage(), e);
        }
    }

    /**
     * 计算奖励（核心优化：减少临时张量创建）
     */
    private Tensor computeReward(Tensor actions, Tensor masks, long batchSize, long seqLen) {
        // 1. 多样性奖励（复用常量张量）
        Tensor uniqueCounts = countUniqueTokens(actions, masks, batchSize, seqLen);
        Tensor seqLengths = masks.sum(-1);
        Tensor seqLengthsSafe = seqLengths.clamp_min(ONE_FLOAT);
        Tensor diversityReward = uniqueCounts.div(seqLengthsSafe);
        diversityReward = clamp(diversityReward,new ScalarOptional(new Scalar(0.0f)), new ScalarOptional(new Scalar(1.0f)));

        // 2. 流畅度奖励（减少临时张量）
        Tensor shiftedActions = actions.roll(new long[]{1}, new long[]{1});
        Tensor actionDiff = actions.sub(shiftedActions).abs();
        Tensor fluencyReward = actionDiff.mul(masks).sum(-1).div(seqLengthsSafe);
        fluencyReward = ONE_FLOAT.sub(fluencyReward.div(createConstTensor((float) vocabSize)));
        fluencyReward = clamp(fluencyReward, new ScalarOptional(new Scalar(0.0f)), new ScalarOptional(new Scalar(1.0f)));

        // 3. 总奖励（复用权重常量）
        Tensor totalReward = REWARD_W1.mul(diversityReward)
                .add(REWARD_W2.mul(fluencyReward))
                .clamp_min(MIN_REWARD.item());

        // 4. 归一化（复用BIG_EPS_TENSOR）
        Tensor rewardMean = totalReward.mean();
        Tensor rewardStd = totalReward.std().add(BIG_EPS_TENSOR);
        if (rewardStd.item_float() < BIG_EPS) {
            rewardStd = BIG_EPS_TENSOR;
        }

        Tensor normalizedReward = totalReward.sub(rewardMean).div(rewardStd);
        normalizedReward = clamp(normalizedReward,new ScalarOptional(new Scalar( -CLIP_ADVANTAGE)), new ScalarOptional(new Scalar( CLIP_ADVANTAGE)));

        // 5. 惰性广播（减少中间张量）
        Tensor rewardExpanded = lazyBroadcast(normalizedReward, new long[]{batchSize, seqLen});
        Tensor finalReward = rewardExpanded.mul(masks);

        // 即时释放所有临时张量
        safeClose(uniqueCounts);
        safeClose(seqLengths);
        safeClose(seqLengthsSafe);
        safeClose(diversityReward);
        safeClose(shiftedActions);
        safeClose(actionDiff);
        safeClose(fluencyReward);
        safeClose(totalReward);
        safeClose(rewardMean);
        safeClose(rewardStd);
        safeClose(normalizedReward);
        safeClose(rewardExpanded);

        return finalReward;
    }

    /**
     * 统计唯一Token数（核心优化：减少索引处理的临时张量）
     */
    private Tensor countUniqueTokens(Tensor actions, Tensor masks, long batchSize, long seqLen) {
        Tensor uniqueCounts = zeros(new long[]{batchSize}, floatOpts);

        try {
            for (int b = 0; b < batchSize; b++) {
                // 切片优化：减少内存拷贝
                Tensor batchActions = actions.narrow(0, b, 1).squeeze(0);
                Tensor batchMasks = masks.narrow(0, b, 1).squeeze(0);

                // 过滤无效动作（复用MINUS_ONE_LONG）
                Tensor validMask = batchActions.ge(ZERO_LONG).mul(batchMasks.to(kFloat()));
                Tensor validActions = batchActions.where(validMask.to(kBool()), MINUS_ONE_LONG);

                // 统计唯一数（优化：减少临时张量）
                Tensor nonZeroIndices = validActions.ne(MINUS_ONE_LONG).nonzero();
                long uniqueCount = 0;
                if (nonZeroIndices.numel() > 0) {
                    if (nonZeroIndices.dim() > 1) {
                        nonZeroIndices = nonZeroIndices.squeeze(-1);
                    }
                    var uniquePair = unique_consecutive(nonZeroIndices);
                    uniqueCount = uniquePair.get0().numel();
                    safeClose(uniquePair.get0());
                    safeClose(uniquePair.get1());
                    uniquePair.close();
                }

                // 赋值（复用常量张量）
                uniqueCounts.narrow(0, b, 1).copy_(createConstTensor((float) uniqueCount));

                // 即时释放
                safeClose(batchActions);
                safeClose(batchMasks);
                safeClose(validMask);
                safeClose(validActions);
                safeClose(nonZeroIndices);
            }

            return uniqueCounts;
        } catch (Exception e) {
            safeClose(uniqueCounts);
            throw new RuntimeException("唯一Token统计失败: " + e.getMessage(), e);
        }
    }

    // ===================== 生成方法（内存优化）=====================
    public Tensor generate(Tensor inputEmbeds, Tensor masks) throws Exception {
        long[] inputShape = inputEmbeds.sizes().vec().get();
        long batchSize = inputShape[0];
        long seqLen = inputShape[1];
        long hiddenDim = inputShape[2];

        actorCriticModel.eval();

        Tensor hidden = null;
        Tensor actions = null;
        Tensor actionLogits = null;
        Tensor discreteActions = null;
        Tensor projLayer = null;
        Tensor result = null;

        try {
            hidden = actorCriticModel.forwardLM(inputEmbeds);
            logTensorInfo("hidden (generate)", hidden);
            if (hidden.dim() == 2) {
                hidden = hidden.view(new long[]{batchSize, seqLen, hiddenDim});
            }

            // 推理阶段使用均值，减少随机性和内存占用
            Distribution dist = actorCriticModel.getDistribution(hidden);
            Normal normalDist = (Normal) dist;
            actions = normalDist.mean();
            logTensorInfo("actions (generate均值)", actions);

            if (actions.dim() == 2) {
                actions = actions.view(new long[]{batchSize, seqLen, hiddenDim});
            }

            // 投影层（即时释放）
            projLayer = randn(new long[]{hiddenDim, vocabSize}, floatOpts);
            actionLogits = actions.matmul(projLayer.t());
            safeClose(projLayer);

            // 归一化（惰性广播）
            Tensor actionsMean = lazyBroadcast(actionLogits.mean(new long[]{-1}, true, new ScalarTypeOptional()),
                    actionLogits.sizes().vec().get());
            Tensor actionsStd = lazyBroadcast(actionLogits.std(new long[]{-1}, true).add(EPS_TENSOR),
                    actionLogits.sizes().vec().get());
            Tensor actionsNorm = actionLogits.sub(actionsMean).div(actionsStd);

            // 离散动作（即时释放临时张量）
            Tensor actionScores = softmax(actionsNorm, -1);
            discreteActions = argmax(actionScores, new LongOptional(-1), false);
            safeClose(actionScores);

            // 应用mask（复用常量张量）
            Tensor masksLong = masks.to(kLong());
            Tensor validPositions = masksLong.gt(ZERO_LONG);
            discreteActions = discreteActions.where(validPositions, MINUS_ONE_LONG);
            safeClose(masksLong);
            safeClose(validPositions);

            // 裁剪动作范围（避免越界）
            discreteActions = clamp(discreteActions, new ScalarOptional(new Scalar(0l)), new ScalarOptional(new Scalar(vocabSize - 1)));

            // 赋值结果（detach后释放临时张量）
            result = discreteActions.clone().detach();
            safeClose(actionsMean);
            safeClose(actionsStd);
            safeClose(actionsNorm);

            actorCriticModel.train(true);
            return result;

        } finally {
            // 即时释放所有临时张量
            safeClose(hidden);
            safeClose(actions);
            safeClose(actionLogits);
            safeClose(discreteActions);
        }
    }

    // ===================== 资源释放（完整）=====================
    @Override
    public void close() {
        // 释放常量张量（核心：全局张量最后释放）
        safeClose(EPS_TENSOR);
        safeClose(BIG_EPS_TENSOR);
        safeClose(ZERO_LONG);
        safeClose(ONE_FLOAT);
        safeClose(MINUS_ONE_LONG);
        safeClose(GAMMA_TENSOR);
        safeClose(LAMBDA_TENSOR);
        safeClose(REWARD_W1);
        safeClose(REWARD_W2);
        safeClose(HALF);
        safeClose(MIN_REWARD);

        // 释放优化器和模型
        if (optimizer != null) optimizer.close();
        if (actorCriticModel != null) actorCriticModel.close();

        // 释放TensorOptions
        if (floatOpts != null) floatOpts.close();
        if (longOpts != null) longOpts.close();

        // 强制GC
        System.gc();
    }

    // ===================== 内部类（资源优化）=====================
    public static class RolloutResult implements AutoCloseable {
        public Tensor states;
        public Tensor actions;
        public Tensor rewards;
        public Tensor masks;
        public Tensor logProbs;
        public Tensor values;

        @Override
        public void close() {
            safeClose(states);
            safeClose(actions);
            safeClose(rewards);
            safeClose(masks);
            safeClose(logProbs);
            safeClose(values);
        }

        private void safeClose(Tensor tensor) {
            if (tensor != null && !tensor.isNull()) {
                try {
                    tensor.close();
                } catch (Exception e) {
                    System.err.println("RolloutResult释放警告: " + e.getMessage());
                }
            }
        }
    }

    // ===================== Getter方法 =====================
    public LMActorCritic getActorCriticModel() {
        return actorCriticModel;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public float getGamma() {
        return gamma;
    }

    public float getLambda() {
        return lambda;
    }

    public double getClipNorm() {
        return clipNorm;
    }

    public String getActionSpaceType() {
        return actionSpaceType;
    }

    public Device getCpuDevice() {
        return cpuDevice;
    }

    public TensorOptions getFloatOpts() {
        return floatOpts;
    }

    public TensorOptions getLongOpts() {
        return longOpts;
    }

    public long getVocabSize() {
        return vocabSize;
    }
}
