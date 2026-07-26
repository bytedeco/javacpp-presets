package org.bytedeco.pytorch.rl;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;

public class RLMonitor {
    /**
     * 计算 KL 散度：E[log(p/q)] = log_p - log_q
     */
    public static float calculateKL(Tensor oldLogProbs, Tensor newLogProbs) {
        try (PointerScope scope = new PointerScope()) {
            Tensor kl = oldLogProbs.sub(newLogProbs).mean();
            return kl.item_float();
        }
    }
}