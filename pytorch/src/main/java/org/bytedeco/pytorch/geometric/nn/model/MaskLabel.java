package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;

public class MaskLabel extends Module {
    private EmbeddingImpl labelEmb;
    private Tensor maskToken;  //new Parameter(
    private long method; // 0: concat, 1: add

    /**
     * @param numClasses 类别数
     * @param outChannels 标签嵌入维度
     * @param method 0 for concat, 1 for add
     */
    public MaskLabel(long numClasses, long outChannels, int method) {
        this.method = method;
        this.labelEmb = new EmbeddingImpl(numClasses, outChannels);
        this.maskToken = torch.randn(new long[]{1, outChannels});

        register_module("labelEmb", labelEmb);
        register_parameter("maskToken", maskToken);
    }

    /**
     * @param x 节点特征 [N, F]
     * @param y 节点标签 [N]
     * @param mask Boolean Mask [N] (True 表示标签可见/用于训练，False 表示需要预测/被Mask)
     */
    public Tensor forward(Tensor x, Tensor y, Tensor mask) {
        // 1. 获取所有节点的 Label Embedding
        // 注意：这里可能会获取到无效标签的 Embedding，但后续会被覆盖
        Tensor score = labelEmb.forward(y);

        // 2. 对于 Mask 为 False (不可见) 的节点，替换为 MaskToken
        // maskToken expanded: [N, C]
        Tensor maskExpanded = mask.unsqueeze(1).expand_as(score);

        // out[i] = mask[i] ? score[i] : maskToken
        // torch.where(condition, x, y) -> condition true yield x, else y
        Tensor labelFeat = torch.where(maskExpanded, score, maskToken.expand_as(score));

        // 3. 融合
        if (method == 0) {
            // Concat
            return torch.cat(new TensorVector(x, labelFeat), 1);
        } else {
            // Add (前提是维度一致)
            return x.add(labelFeat);
        }
    }
}
