package org.bytedeco.pytorch.rl;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.scalar_tensor;
import static org.bytedeco.pytorch.global.torch.zeros_like;

public class GAE {
    /**
     * @param rewards [T, B], values [T, B], masks [T, B]
     * @return Tensor[] {Advantages, Returns}
     */
    /**
     * @param rewards [T, B]
     * @param values  [T, B] or [T+1, B] (if T+1, bootstrap uses values[T])
     * @param masks   [T, B]  1 = not done / continue, 0 = episode end
     * @return Tensor[] {advantages [T,B], returns [T,B]}
     */
    public static Tensor[] compute(Tensor rewards, Tensor values, Tensor masks, float gamma, float tau) {
        long T = rewards.size(0);
        long valueLen = values.size(0);
        if (valueLen != T && valueLen != T + 1) {
            throw new IllegalArgumentException(
                    "values length must be T or T+1 (got values=" + valueLen + ", T=" + T + ")");
        }
        Tensor advantages = zeros_like(rewards);
        Tensor lastGae = scalar_tensor(new Scalar(0.0), rewards.options());

        // Reverse recursion t = T-1 .. 0
        for (long t = T - 1; t >= 0; t--) {
            // Bootstrap: values[T] if provided, else 0 at terminal of rollout
            Tensor nextVal;
            if (valueLen == T + 1) {
                nextVal = values.select(0, t + 1);
            } else if (t == T - 1) {
                nextVal = zeros_like(values.select(0, t));
            } else {
                nextVal = values.select(0, t + 1);
            }
            // TD-Error: delta = r + gamma * V(s') * mask - V(s)
            Tensor delta = rewards.select(0, t)
                    .add(nextVal.mul(new Scalar(gamma)).mul(masks.select(0, t)))
                    .sub(values.select(0, t));

            // GAE = delta + gamma * tau * mask * lastGae
            lastGae = delta.add(lastGae.mul(new Scalar(gamma * tau)).mul(masks.select(0, t)));
            advantages.select(0, t).copy_(lastGae);
        }
        // returns use only the first T value estimates
        Tensor valueSlice = valueLen == T + 1
                ? values.slice(0, new LongOptional(0), new LongOptional(T), 1)
                : values;
        Tensor returns = advantages.add(valueSlice);
        return new Tensor[]{advantages, returns};
    }
}