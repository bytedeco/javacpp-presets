package org.bytedeco.pytorch.geometric.attention;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.enumtype.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.geometric.utils.AttentionUtils;

/**
 * Polynormer Attention (Linear Attention)
 * 使用 phi(x) = elu(x) + 1 作为核函数
 */
public class PolynormerAttention extends Module {
    private LinearImpl linQ, linK, linV, linOut;
    private long numHeads;
    private long headDim;

    public PolynormerAttention(long inChannels, long numHeads) {
        super();
        this.numHeads = numHeads;
        this.headDim = inChannels / numHeads;

        this.linQ = new LinearImpl(inChannels, inChannels);
        this.linK = new LinearImpl(inChannels, inChannels);
        this.linV = new LinearImpl(inChannels, inChannels);
        this.linOut = new LinearImpl(inChannels, inChannels);

        register_module("linQ", linQ);
        register_module("linK", linK);
        register_module("linV", linV);
        register_module("linOut", linOut);
    }

    public Tensor forward(Tensor x) {
        long N = x.size(0);
        long C = x.size(1);

        // 1. Projections [N, H, D]
        Tensor q = linQ.forward(x).view(N, numHeads, headDim);
        Tensor k = linK.forward(x).view(N, numHeads, headDim);
        Tensor v = linV.forward(x).view(N, numHeads, headDim);

        // 2. Kernel Map: elu(x) + 1
        Tensor qPrime = AttentionUtils.kernel_elu(q);
        Tensor kPrime = AttentionUtils.kernel_elu(k);

        // 3. Linear Attention Logic
        // Einsum('nhd, nhm -> hdm') (KV Global)
        // Permute to [H, N, D]
        Tensor qP = qPrime.permute(1, 0, 2); // [H, N, D]
        Tensor kP = kPrime.permute(1, 0, 2); // [H, N, D]
        Tensor vP = v.permute(1, 0, 2);      // [H, N, D]

        // KV = K^T V -> [H, D, N] @ [H, N, D] -> [H, D, D]
        Tensor kPt = kP.permute(0, 2, 1);
        Tensor kv = kPt.matmul(vP);

        // 4. Numerator: Q' (K'^T V)
        // [H, N, D] @ [H, D, D] -> [H, N, D]
        Tensor out = qP.matmul(kv);

        // 5. Denominator: Q' (K'^T 1)
        // K_sum = sum(K', dim=1) -> [H, D]
        Tensor kSum = kP.sum(new long[]{1}, false, new ScalarTypeOptional()).unsqueeze(2); // [H, D, 1]

        // Norm = Q' @ kSum -> [H, N, D] @ [H, D, 1] -> [H, N, 1]
        Tensor norm = qP.matmul(kSum);
        norm = norm.add(new Scalar(1e-6)); // Prevent div 0

        // 6. Normalize
        out = out.div(norm);

        // 7. Output
        out = out.permute(1, 0, 2).reshape(N, C);
        return linOut.forward(out);
    }
}