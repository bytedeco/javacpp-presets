package org.bytedeco.pytorch.rl.loss;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.log_sigmoid;
public class DPOLoss {
    public Tensor computeLoss(Tensor policyLogProbs, Tensor refLogProbs, boolean preferred) {
        // 公式: log_sigmoid(beta * ((log_p_w - log_ref_w) - (log_p_l - log_ref_l)))
        float beta = 0.1f;
        Tensor diff = policyLogProbs.sub(refLogProbs).mul(new Scalar(beta));
        // 此处逻辑取决于胜出(preferred)和失败的对比
        return log_sigmoid(diff).neg().mean();
    }
}
