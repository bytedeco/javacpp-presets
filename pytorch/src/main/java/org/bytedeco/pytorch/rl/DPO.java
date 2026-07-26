package org.bytedeco.pytorch.rl;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.log_sigmoid;

public class DPO {
    private final float beta;

    public DPO(float beta) { this.beta = beta; }

    /**
     * DPO 损失计算
     * @param policyLogProbsChosen   当前策略对胜出样本的 logProb
     * @param policyLogProbsRejected 当前策略对失败样本的 logProb
     * @param refLogProbsChosen      参考模型对胜出样本的 logProb
     * @param refLogProbsRejected     参考模型对失败样本的 logProb
     */
    public Tensor computeLoss(Tensor policyLogProbsChosen, Tensor policyLogProbsRejected,
                              Tensor refLogProbsChosen, Tensor refLogProbsRejected) {

        // 计算当前策略相对于参考模型的 log-ratio
        Tensor chosenLogratios = policyLogProbsChosen.sub(refLogProbsChosen);
        Tensor rejectedLogratios = policyLogProbsRejected.sub(refLogProbsRejected);

        // DPO 核心公式: -E[log_sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))]
        Tensor logits = chosenLogratios.sub(rejectedLogratios).mul(new Scalar(beta));

        // 使用 log_sigmoid 保证数值稳定性
        return log_sigmoid(logits).mean().neg();
    }
    public Tensor computeDPOLoss(Tensor policyLpChosen, Tensor policyLpRejected,
                                 Tensor refLpChosen, Tensor refLpRejected, float beta) {
        Tensor chosenRatio = policyLpChosen.sub(refLpChosen);
        Tensor rejectedRatio = policyLpRejected.sub(refLpRejected);

        Tensor logits = chosenRatio.sub(rejectedRatio).mul(new Scalar(beta));
        return log_sigmoid(logits).mean().neg();
    }
}