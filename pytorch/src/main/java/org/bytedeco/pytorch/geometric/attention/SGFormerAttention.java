package org.bytedeco.pytorch.geometric.attention;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

/**
 * SGFormer Attention (Simple Global Attention)
 * 通过 Global Average Pooling 提取全局信息，然后广播回所有节点。
 * 极简、极快、线性复杂度。
 */
public class SGFormerAttention extends Module {
    private LinearImpl linQ, linK, linV, linOut;
    private long numHeads;
    private long headDim;

    public SGFormerAttention(long inChannels, long numHeads) {
        super();
        this.numHeads = numHeads;
        this.headDim = inChannels / numHeads;

        // 这里我们不需要 Full Attention Matrix
        // 只需要 Q 投影 X，K, V 投影 Global Node
        this.linQ = new LinearImpl(inChannels, inChannels);
        this.linK = new LinearImpl(inChannels, inChannels); // Input 1 (Global) -> Output In
        this.linV = new LinearImpl(inChannels, inChannels); // Input 1 (Global) -> Output In
        this.linOut = new LinearImpl(inChannels, inChannels);

        register_module("linQ", linQ);
        register_module("linK", linK);
        register_module("linV", linV);
        register_module("linOut", linOut);
    }

    public Tensor forward(Tensor x) {
        // x: [N, C]
        long N = x.size(0);
        long C = x.size(1);

        // 1. Calculate Global Node g = Mean(x) -> [1, C]
        Tensor g = x.mean(new long[]{0}, true, new ScalarTypeOptional());

        // 2. Projections
        // Q from all nodes: [N, H, D]
        Tensor q = linQ.forward(x).view(N, numHeads, headDim);

        // K, V from global node: [1, H, D]
        Tensor k = linK.forward(g).view(1, numHeads, headDim);
        Tensor v = linV.forward(g).view(1, numHeads, headDim);

        // 3. Simple Attention: Q * K^T * V
        // 注意：这里 K 只有一个 token，所以 attention score 是 [N, 1]
        // Score = (Q * K) / sqrt(d)
        // [N, H, D] * [1, H, D] -> [N, H, D] -> sum(D) -> [N, H, 1]
        Tensor score = q.mul(k).sum(new long[]{2}, true, new ScalarTypeOptional());
        score = score.mul(new Scalar(1.0 / Math.sqrt(headDim)));

        // Softmax over global tokens (只有1个token，softmax必定是1.0)
        // 但 SGFormer 原文其实不需要 softmax，或者说这就是 scaling
        // 这里为了保持 Attention 语义，我们保留 score，或者对其做 sigmoid/relu
        // SGFormer 实际上使用的是 Input-dependent scaling
        // 我们这里使用 Sigmoid 作为 Gating
        Tensor attn = torch.sigmoid(score); // [N, H, 1]

        // 4. Output = Attn * V
        // [N, H, 1] * [1, H, D] -> [N, H, D]
        Tensor out = v.mul(attn);

        // 5. Final
        out = out.reshape(N, C);
        return linOut.forward(out);
    }
}
