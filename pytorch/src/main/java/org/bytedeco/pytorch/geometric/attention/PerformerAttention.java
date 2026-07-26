package org.bytedeco.pytorch.geometric.attention;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.enumtype.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.geometric.utils.AttentionUtils;

/**
 * Performer Attention (FAVOR+)
 * 线性复杂度 O(N * D * M)
 */
public class PerformerAttention extends Module {
    private LinearImpl linQ, linK, linV, linOut;
    private Tensor projectionMatrix; // 随机投影矩阵 (Buffer, 不参与梯度更新)
    private long numHeads;
    private long headDim;
    private long numFeatures; // 随机特征数量 (m)

    public PerformerAttention(long inChannels, long numHeads, long numFeatures) {
        super();
        this.numHeads = numHeads;
        this.headDim = inChannels / numHeads;
        this.numFeatures = numFeatures;

        this.linQ = new LinearImpl(inChannels, inChannels);
        this.linK = new LinearImpl(inChannels, inChannels);
        this.linV = new LinearImpl(inChannels, inChannels);
        this.linOut = new LinearImpl(inChannels, inChannels);

        register_module("linQ", linQ);
        register_module("linK", linK);
        register_module("linV", linV);
        register_module("linOut", linOut);

        // 初始化随机投影矩阵 [HeadDim, M]
        // 注册为 buffer，这样 to(device) 时会一起移动
        Tensor proj = AttentionUtils.create_projection_matrix(numFeatures, headDim, true);
        this.projectionMatrix = proj;
        register_buffer("projectionMatrix", projectionMatrix);
    }

    public Tensor forward(Tensor x) {
        long N = x.size(0);
        long C = x.size(1);

        // 1. Projections [N, Heads, Dim]
        Tensor q = linQ.forward(x).view(N, numHeads, headDim);
        Tensor k = linK.forward(x).view(N, numHeads, headDim);
        Tensor v = linV.forward(x).view(N, numHeads, headDim);

        // 2. Kernel Feature Map phi(.)
        // 由于 Performer Kernel 通常针对每个 Head 独立投影，或者共享投影
        // 这里简化为所有 Head 共享投影矩阵

        // Q, K: [N, H, D] -> Reshape -> [N*H, D]
        Tensor qFlat = q.reshape(N * numHeads, headDim);
        Tensor kFlat = k.reshape(N * numHeads, headDim);

        // Apply Kernel: [N*H, M]
        Tensor qPrime = AttentionUtils.kernel_performer(qFlat, projectionMatrix, true);
        Tensor kPrime = AttentionUtils.kernel_performer(kFlat, projectionMatrix, false);

        // Reshape back: [N, H, M]
        qPrime = qPrime.view(N, numHeads, numFeatures);
        kPrime = kPrime.view(N, numHeads, numFeatures);

        // 3. Efficient Attention: Q' (K'^T V)
        // K': [N, H, M], V: [N, H, D]
        // K'^T V -> Einsum('nhm, nhd -> hmd')
        // JavaCPP 中用 permute + matmul 模拟

        // kPrime: [H, N, M]
        Tensor kPrimeT = kPrime.permute(1, 0, 2); // [H, N, M] -> transpose N,M for matmul -> [H, M, N]
        Tensor kPrimeTrans = kPrime.permute(1, 2, 0); // [H, M, N]

        Tensor vPerm = v.permute(1, 0, 2); // [H, N, D]

        // KV_Global = [H, M, N] @ [H, N, D] -> [H, M, D]
        Tensor kvGlobal = kPrimeTrans.matmul(vPerm);

        // 4. Denominator (Normalization)
        // D = Q' (K'^T 1)
        Tensor kSum = kPrimeTrans.sum(new long[]{2}, true,new ScalarTypeOptional()); // [H, M, 1]
        // Q': [H, N, M]
        Tensor qPrimePerm = qPrime.permute(1, 0, 2); // [H, N, M]

        // Norm = [H, N, M] @ [H, M, 1] -> [H, N, 1]
        Tensor norm = qPrimePerm.matmul(kSum);
        // Prevent div 0
        norm = norm.add(new Scalar(1e-6));

        // 5. Numerator
        // Out = [H, N, M] @ [H, M, D] -> [H, N, D]
        Tensor out = qPrimePerm.matmul(kvGlobal);

        // 6. Final: Out / Norm
        out = out.div(norm);

        // [H, N, D] -> [N, H, D] -> [N, C]
        out = out.permute(1, 0, 2).reshape(N, C);

        return linOut.forward(out);
    }
}