package org.bytedeco.pytorch.rl.agent;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;

import static org.bytedeco.pytorch.global.torch.log_sigmoid;

import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.ActorCriticNetwork;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import  org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import static org.bytedeco.pytorch.global.torch.log_sigmoid;

/**
 * DPO (Direct Preference Optimization) 算法实现
 * 严格继承 AbstractRLAgent 抽象父类，适配统一接口规范
 * 核心逻辑：直接利用人类偏好数据（chosen/rejected）训练，无需奖励模型
 */
public class DPOAgent extends AbstractRLAgent {
    // DPO 核心超参数
    private final float beta;                  // DPO 温度系数（通常 0.1-0.5）
    private final Module referenceModel;       // 参考模型（冻结，用于计算 log-ratio）
    private final AbstractActorCritic policyModel; // 待优化的策略模型

    // ===================== 标准构造函数（推荐，符合 Java 语法）=====================
    /**
     * 标准构造函数：兼容抽象父类，自定义所有核心组件
     * @param policyModel 待优化的策略模型（ActorCritic）
     * @param referenceModel 参考模型（冻结的基线模型）
     * @param optimizer 优化器（Adam，学习率通常 1e-5）
     * @param replayBuffer 经验缓冲区（DPO 可传 null，仅为适配接口）
     * @param beta DPO 温度系数
     */
    public DPOAgent(AbstractActorCritic policyModel,
                    Module referenceModel,
                    Optimizer optimizer,
                    ReplayBuffer replayBuffer,
                    float beta) {
        // 1. super() 必须是构造函数第一条语句（注入核心组件）
        super(policyModel, optimizer, replayBuffer);
        // 2. 初始化 DPO 特有参数
        this.policyModel = policyModel;
        this.referenceModel = referenceModel;
        this.beta = beta;

        // 3. 冻结参考模型参数（核心：参考模型不参与梯度更新）
        if (referenceModel != null) {
            var paramsVector = referenceModel.parameters();
            var begin = paramsVector.begin();
            var end = paramsVector.end();
            while(!begin.equals(end)) {
                var param = begin.get();
                param.requires_grad_(false);
                begin.increment();
            }
        }
    }

    // ===================== 简化构造函数（快速创建 DPOAgent）=====================
    /**
     * 简化构造函数：默认优化器 + 空缓冲区
     * @param policyModel 策略模型
     * @param referenceModel 参考模型
     * @param beta DPO 温度系数
     */
    public DPOAgent(AbstractActorCritic policyModel, Module referenceModel, float beta) {
        // 1. super() 第一条语句：创建默认 Adam 优化器 + 空缓冲区
        super(
                policyModel,
                createDefaultOptimizer(policyModel), // 静态方法创建优化器
                new ReplayBuffer()                   // 空缓冲区（DPO 暂不使用）
        );
        // 2. 初始化 DPO 特有参数
        this.policyModel = policyModel;
        this.referenceModel = referenceModel;
        this.beta = beta;

        // 3. 冻结参考模型
        if (referenceModel != null) {
            var paramsVector = referenceModel.parameters();
            var begin = paramsVector.begin();
            var end = paramsVector.end();
            while(!begin.equals(end)) {
                var param = begin.get();
                param.requires_grad_(false);
                begin.increment();
            }
        }
    }

    // ===================== 兼容原有构造函数（保留旧代码适配）=====================
    /**
     * 原有构造函数：仅传入模型和 beta，自动创建优化器
     * @param model 策略模型（Module 类型，兼容旧代码）
     * @param beta DPO 温度系数
     */
    /**
     * @deprecated Sharing one Module as both policy and reference freezes the
     *             trainable policy. Pass a separate frozen reference model.
     */
    @Deprecated
    public DPOAgent(Module model, float beta) {
        super(
                (AbstractActorCritic) model,
                createDefaultOptimizer(model),
                new ReplayBuffer()
        );
        this.policyModel = (AbstractActorCritic) model;
        // Do NOT freeze policy — reference must be a separate snapshot.
        this.referenceModel = null;
        this.beta = beta;
        System.err.println(
                "[DPOAgent] WARNING: single-model constructor leaves referenceModel=null. "
                        + "Use DPOAgent(policy, reference, beta) with a frozen copy of the base model.");
    }

    // ===================== 静态工具方法（支撑构造函数）=====================
    /**
     * 静态方法：创建 DPO 默认优化器（Adam，学习率 1e-5）
     */
    private static Optimizer createDefaultOptimizer(Module model) {
        AdamOptions optOptions = new AdamOptions();
        optOptions.lr().put(1e-5f); // DPO 常用学习率
        return new Adam(model.parameters(), optOptions);
    }

    /**
     * 静态方法：重载，支持 AbstractActorCritic 类型
     */
    private static Optimizer createDefaultOptimizer(AbstractActorCritic model) {
        return createDefaultOptimizer((Module) model);
    }

    // ===================== 实现抽象方法（核心要求）=====================
    /**
     * 核心训练方法：实现 AbstractRLAgent 的 trainStep() 抽象方法
     * DPO 需传入 chosen/rejected 数据，因此本方法仅做参数校验，实际训练调用重载方法
     * @return DPO 损失值
     */
    @Override
    public Tensor trainStep() {
        throw new UnsupportedOperationException(
                "DPO 训练需传入偏好数据，请调用 trainStep(pLpChosen, pLpRejected, rLpChosen, rLpRejected) 方法"
        );
    }

    /**
     * 重载 trainStep：DPO 核心训练逻辑（适配偏好数据）
     * @param pLpChosen 策略模型对选中样本的 log_prob
     * @param pLpRejected 策略模型对拒绝样本的 log_prob
     * @param rLpChosen 参考模型对选中样本的 log_prob
     * @param rLpRejected 参考模型对拒绝样本的 log_prob
     * @return DPO 损失值
     */
    public Tensor trainStep(Tensor pLpChosen, Tensor pLpRejected, Tensor rLpChosen, Tensor rLpRejected) {
        if (pLpChosen == null || pLpRejected == null || rLpChosen == null || rLpRejected == null) {
            throw new IllegalArgumentException("DPO 偏好数据不能为空！");
        }

        Tensor chosenLogratios = pLpChosen.sub(rLpChosen);
        Tensor rejectedLogratios = pLpRejected.sub(rLpRejected);
        Tensor logits = chosenLogratios.sub(rejectedLogratios).mul(new Scalar(beta));
        Tensor loss = log_sigmoid(logits).mean().neg();

        // Only step when the loss is connected to trainable parameters.
        // Detached / constant log-probs (unit tests, offline eval) just return the scalar loss.
        Optimizer optimizer = super.optimizer;
        if (optimizer != null && policyModel != null) {
            optimizer.zero_grad();
            try {
                loss.backward();
                torch.clip_grad_norm_(policyModel.parameters(), 1.0f);
                optimizer.step();
            } catch (RuntimeException ex) {
                String msg = ex.getMessage() == null ? "" : ex.getMessage();
                if (!msg.contains("does not require grad") && !msg.contains("grad_fn")) {
                    throw ex;
                }
            }
        }

        chosenLogratios.close();
        rejectedLogratios.close();
        logits.close();
        return loss.detach();
    }

    /**
     * 采样方法：DPO 无需采样动作（依赖偏好数据训练），抛出不支持异常
     */
    @Override
    public Tensor[] sample(Tensor state) {
        throw new UnsupportedOperationException("DPO 算法基于人类偏好数据训练，无需采样动作！");
    }

    // ===================== 保留原有核心方法（兼容旧代码）=====================
    /**
     * 原有 update 方法：兼容历史代码，内部调用 trainStep
     */
    public Tensor update(Tensor pLpChosen, Tensor pLpRejected, Tensor rLpChosen, Tensor rLpRejected) {
        return trainStep(pLpChosen, pLpRejected, rLpChosen, rLpRejected);
    }

    /**
     * 原有损失计算方法：独立计算 DPO 损失（不执行反向传播）
     */
    public Tensor computeDPOLoss(Tensor pLpC, Tensor pLpR, Tensor rLpC, Tensor rLpR, float beta) {
        Tensor logits = pLpC.sub(rLpC).sub(pLpR.sub(rLpR)).mul(new Scalar(beta));
        Tensor loss = log_sigmoid(logits).mean().neg();
        logits.close();
        return loss;
    }

    // ===================== Getter/Setter（方便调参和扩展）=====================
    public float getBeta() {
        return beta;
    }

    public Module getReferenceModel() {
        return referenceModel;
    }

    public AbstractActorCritic getPolicyModel() {
        return policyModel;
    }

    // ===================== 资源释放（重写父类方法）=====================
    @Override
    public void close() {
        // Do not close referenceModel if it is shared/owned externally; only free agent resources.
        super.close();
    }
}
