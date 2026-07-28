package org.bytedeco.pytorch.rl.agent;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.rl.critic.LMActorCritic;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * DPO训练器（无Scalar版）
 * 1. 完全移除所有Scalar相关代码
 * 2. 严格使用基础类型创建tensor
 * 3. 匹配所有提供的API签名
 */
public class LMDPOAgent implements AutoCloseable {
    private final LMActorCritic policyModel;
    private final LMActorCritic refModel;
    private final Optimizer optimizer;
    private final float beta;
    private final float clipNorm;
    private final String actionSpaceType;

    // 合规常量Tensor（无Scalar）
    private final Tensor BETA_TENSOR;
    private final Tensor EPS_TENSOR;
    private final Tensor ZERO_TENSOR_LONG;
    private final Tensor VOCAB_TENSOR_LONG;

    /**
     * 构造函数（严格匹配API，无Scalar）
     */
    public LMDPOAgent(LMActorCritic policyModel, LMActorCritic refModel,
                      float beta, float lr, float clipNorm, String actionSpaceType,
                      TensorOptions floatOpts, Device cpuDevice) {
        this.policyModel = policyModel;
        this.refModel = refModel;
        this.beta = beta;
        this.clipNorm = clipNorm;
        this.actionSpaceType = actionSpaceType;

        // 1. 冻结参考模型参数
        freezeModel(refModel);

        // 2. 初始化优化器（合规）
        AdamOptions optOpts = new AdamOptions();
        optOpts.lr().put(lr);
        optOpts.weight_decay().put(0.01f);
        optOpts.betas().put(new double[]{0.9, 0.999});
        this.optimizer = new Adam(policyModel.parameters(), optOpts);

        // 3. 预创建常量Tensor（无Scalar，严格使用基础类型）
        long vocabSize = 30522L;
        TensorOptions longOpts = new TensorOptions()
                .device(new DeviceOptional(cpuDevice))
                .dtype(new ScalarTypeOptional(kLong()));

        // 严格使用float/long+TensorOptions创建tensor（无Scalar）
        this.BETA_TENSOR = tensor(beta, floatOpts)
                .to(new Device(cpuDevice), kFloat()).clone().detach();
        this.EPS_TENSOR = tensor(1e-8f, floatOpts)
                .to(new Device(cpuDevice), kFloat()).clone().detach();
        this.ZERO_TENSOR_LONG = tensor(0L, longOpts);
        this.VOCAB_TENSOR_LONG = tensor(vocabSize, longOpts);
    }

    /**
     * 单步训练（核心DPO逻辑，无Scalar）
     */
    public Tensor trainStep(Tensor inputEmbeds, Tensor chosenActions,
                            Tensor rejectedActions, Tensor masks) {
        try (
                // 前向传播
                Tensor policyHidden = policyModel.forwardLM(inputEmbeds);
                Tensor refHidden = refModel.forwardLM(inputEmbeds);
                Distribution policyDist = policyModel.getDistribution(policyHidden);
                Distribution refDist = refModel.getDistribution(refHidden)
        ) {
            try (
                    // 计算对数概率（合规：掩码相乘）
                    Tensor pLpChosen = policyDist.log_prob(chosenActions).mul(masks).sum(-1);
                    Tensor pLpRejected = policyDist.log_prob(rejectedActions).mul(masks).sum(-1);
                    Tensor rLpChosen = refDist.log_prob(chosenActions).mul(masks).sum(-1);
                    Tensor rLpRejected = refDist.log_prob(rejectedActions).mul(masks).sum(-1);

                    // DPO核心计算（无Scalar，Tensor算术）
                    Tensor chosenRatio = pLpChosen.sub(rLpChosen);
                    Tensor rejectedRatio = pLpRejected.sub(rLpRejected);
                    Tensor logits = chosenRatio.sub(rejectedRatio).mul(BETA_TENSOR);

                    // 损失计算（数值稳定，无Scalar）
                    Tensor dpoLoss = log_sigmoid(logits).mean().neg();
                    Tensor loss = dpoLoss.add(EPS_TENSOR);
            ) {
                // 反向传播
                optimizer.zero_grad();
                loss.backward();
                clip_grad_norm_(policyModel.parameters(), clipNorm);
                optimizer.step();

                return loss.detach().clone();
            }
        } catch (Exception e) {
            throw new RuntimeException("DPO训练步骤失败: " + e.getMessage(), e);
        }
    }

    /**
     * 生成动作（无Scalar，合规）
     */
    public Tensor generate(Tensor inputEmbeds, Tensor masks) throws Exception {
        policyModel.eval();
        try (
                Tensor hidden = policyModel.forwardLM(inputEmbeds);
                Distribution dist = policyModel.getDistribution(hidden);
                // 掩码类型转换（合规，无Scalar）
                Tensor masksLong = masks.to(new Device(kCPU()), kLong()).clone().detach()
        ) {
            Tensor actions = dist.sample().mul(masksLong);
            // 合规clamp（无Scalar，使用tensor.item()创建ScalarOptional）
            actions = clamp(actions,
                    new ScalarOptional(ZERO_TENSOR_LONG.item()),
                    new ScalarOptional(VOCAB_TENSOR_LONG.item()));
            policyModel.train(true);
            return actions.detach().clone();
        }
    }

    /**
     * 冻结模型参数
     */
    private void freezeModel(LMActorCritic model) {
        TensorVector params = model.parameters();
        var begin = params.begin();
        var end = params.end();
        while(!begin.equals(end)) {
            var param = begin.get();
            param.requires_grad_(false);
            begin.increment();
        }
        params.close();
    }

    @Override
    public void close() {
        // 释放所有资源
        if (BETA_TENSOR != null) BETA_TENSOR.close();
        if (EPS_TENSOR != null) EPS_TENSOR.close();
        if (ZERO_TENSOR_LONG != null) ZERO_TENSOR_LONG.close();
        if (VOCAB_TENSOR_LONG != null) VOCAB_TENSOR_LONG.close();

        if (optimizer != null) optimizer.close();
        if (policyModel != null) policyModel.close();
        if (refModel != null) refModel.close();
    }
}

