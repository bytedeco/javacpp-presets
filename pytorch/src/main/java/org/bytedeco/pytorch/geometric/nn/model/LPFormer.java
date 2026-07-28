package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.attention.PerformerAttention;

// 简略实现框架
public class LPFormer extends Module {
    private PerformerAttention attn; // 使用线性 Attention 提高效率

    public LPFormer(long dim) {
        this.attn = new PerformerAttention(dim, 4, 16);
        register_module("attn", attn);
    }

    public Tensor forward(Tensor z, Tensor srcIdx, Tensor dstIdx) {
        // 1. Get Embeddings
        Tensor zSrc = z.index_select(0, srcIdx);
        Tensor zDst = z.index_select(0, dstIdx);

        // 2. Interaction (Hadamard product or Concatenation)
        // Link Prediction typically uses product or MLP(cat)
        Tensor linkFeat = zSrc.mul(zDst);

        // 3. Score
        return linkFeat.sum(new long[]{1}, false,new ScalarTypeOptional()); // Dot product score
    }
}