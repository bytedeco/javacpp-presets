package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;
public class VGAE extends GAE {
    // Encoder 必须返回 mu 和 logstd
    // 我们假设 Encoder 输出 2*Dim，分割为 mu, logstd

    public VGAE(Module encoder) {
        super(encoder);
    }

    public Tensor reparametrize(Tensor mu, Tensor logStd) {
        if (is_training()) {
            Tensor eps = torch.randn_like(logStd);
            return mu.add(eps.mul(logStd.exp()));
        }
        return mu;
    }

    public Tensor klLoss(Tensor mu, Tensor logStd) {
        // -0.5 * sum(1 + 2*logstd - mu^2 - exp(2*logstd))
        return logStd.mul(new Scalar(2)).add(new Scalar(1)).sub(mu.pow(new Scalar(2))).sub(logStd.mul(new Scalar(2)).exp()).sum().mul(new Scalar(-0.5));
    }
}