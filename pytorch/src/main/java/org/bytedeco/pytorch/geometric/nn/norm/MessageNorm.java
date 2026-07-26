package org.bytedeco.pytorch.geometric.nn.norm;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
/**
 * MessageNorm
 * 归一化消息向量 m，使其模长与节点特征 x 保持一致，并乘以可学习系数。
 */
public class MessageNorm extends Module {
    private Parameter scale; // Learnable scale factor

    public MessageNorm(double initScale) {
        super();
        this.scale = new Parameter(torch.tensor(initScale));
        register_parameter("scale", scale);
    }

    /**
     * @param x 节点自身特征 [N, C]
     * @param msg 聚合后的消息 [N, C]
     */
    public Tensor forward(Tensor x, Tensor msg) {
        // 1. Compute L2 Norm (dim=1)
        // norm_x: [N, 1]
        Tensor normX = x.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);
        Tensor normMsg = msg.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);

        // 2. Normalize Message
        // m' = m * (norm_x / (norm_m + eps))
        Tensor ratio = normX.div(normMsg.add(new Scalar(1e-6)));

        Tensor out = msg.mul(ratio);

        // 3. Apply Learnable Scale
        return out.mul(scale);
    }
}