package org.bytedeco.pytorch.geometric.nn.kge;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

public class RotatE extends KGEModel {
    private double epsilon; // 用于初始化相位

    public RotatE(long numNodes, long numRels, long hiddenChannels, double epsilon) {
        // RotatE 中，hiddenChannels 通常指复数维度的数量，实际参数量是 2倍
        // 这里为了简单，我们假设 hiddenChannels 就是实数维度，切分为 Re/Im
        super(numNodes, numRels, hiddenChannels * 2);
        this.epsilon = epsilon;
        // 节点 Embedding: 实部 + 虚部，所以是 hiddenChannels * 2
        this.nodeEmb = new EmbeddingImpl(numNodes, hiddenChannels * 2);
        // 关系 Embedding: 只需要相位 theta，所以是 hiddenChannels (不乘2)
        this.relEmb = new EmbeddingImpl(numRels, hiddenChannels);
        // 初始化 Relation Embedding 的相位 (Phase) 在 -pi 到 pi
        // uniform_(-epsilon, epsilon)
        torch.uniform_(relEmb.weight(), -epsilon, epsilon);
//        register_module("node_Emb", nodeEmb);
//        register_module("rel_Emb", relEmb);
    }

    @Override
    public Tensor forward(Tensor head, Tensor relation, Tensor tail) {
        // 1. 获取 Embedding
        // h, t: [B, 2 * hiddenChannels]
        // r: [B, hiddenChannels] (相位 theta)
        Tensor h = nodeEmb.forward(head);
        Tensor t = nodeEmb.forward(tail);
        Tensor r = relEmb.forward(relation);

        // 2. 拆分节点为 实部 和 虚部
        TensorVector h_parts = torch.chunk(h, 2, 1);
        Tensor h_re = h_parts.get(0); // [B, hiddenChannels]
        Tensor h_im = h_parts.get(1); // [B, hiddenChannels]

        TensorVector t_parts = torch.chunk(t, 2, 1);
        Tensor t_re = t_parts.get(0);
        Tensor t_im = t_parts.get(1);

        // 3. 关系转换为复数单位圆上的点 (Euler's formula: e^{i theta} = cos theta + i sin theta)
        // 此时 r_re 和 r_im 的形状都是 [B, hiddenChannels]，完美匹配 h_re
        Tensor r_re = torch.cos(r);
        Tensor r_im = torch.sin(r);

        // 4. 复数乘法 (Rotate): (h_re + i h_im) * (r_re + i r_im)
        Tensor rotate_re = h_re.mul(r_re).sub(h_im.mul(r_im));
        Tensor rotate_im = h_re.mul(r_im).add(h_im.mul(r_re));

        // 5. 计算欧氏距离: || (h * r) - t ||
        Tensor diff_re = rotate_re.sub(t_re);
        Tensor diff_im = rotate_im.sub(t_im);

        // 计算平方和再开方
        // [B, hiddenChannels] -> sum dim 1 -> [B]
        Tensor dist = diff_re.pow(new Scalar(2))
                .add(diff_im.pow(new Scalar(2)))
                .sum(new long[]{1}, false, new ScalarTypeOptional())
                .sqrt();

        // 返回负距离作为得分
        return dist.neg();
    }

    //    @Override
    public Tensor forward2(Tensor head, Tensor relation, Tensor tail) {
        // 1. 获取 Embedding [B, 2*D]
        Tensor h = nodeEmb.forward(head);
        Tensor t = nodeEmb.forward(tail);
        Tensor r = relEmb.forward(relation);

        // 2. 将 h, t, r 拆分为 Re 和 Im
        TensorVector h_parts = torch.chunk(h, 2, 1);
        Tensor h_re = h_parts.get(0);
        Tensor h_im = h_parts.get(1);

        TensorVector t_parts = torch.chunk(t, 2, 1);
        Tensor t_re = t_parts.get(0);
        Tensor t_im = t_parts.get(1);

        // 3. 约束 r 的模长为 1 (Rotation)
        // r 存储的是相位 theta
        // r_re = cos(theta), r_im = sin(theta)
        // 注意：RotatE 的实现中，relation embedding 初始是 theta，需要 cos/sin 变换
        // 这里假设 relEmb 存储的就是 theta (需要限制范围)
        Tensor r_phase = r.div(new Scalar(epsilon / Math.PI)); // scale to roughly -pi, pi?
        // 标准实现通常直接取 cos, sin
        Tensor r_re = torch.cos(r);
        Tensor r_im = torch.sin(r);

        // 4. Rotate: h * r (Complex Multiplication)
        // (h_re + i h_im) * (r_re + i r_im)
        // = (h_re r_re - h_im r_im) + i (h_re r_im + h_im r_re)
        Tensor rotate_re = h_re.mul(r_re).sub(h_im.mul(r_im));
        Tensor rotate_im = h_re.mul(r_im).add(h_im.mul(r_re));

        // 5. Score: - || rotate - t ||
        // re_diff^2 + im_diff^2
        Tensor diff_re = rotate_re.sub(t_re);
        Tensor diff_im = rotate_im.sub(t_im);

        Tensor dist = diff_re.pow(new Scalar(2)).add(diff_im.pow(new Scalar(2))).sum(new long[]{1}, false, new ScalarTypeOptional()).sqrt();

        // 加上 Margin Loss 的话，通常也需要 Margin
        // RotatE 论文中通常还有 Self-Adversarial Negative Sampling，这里仅实现核心 Score
        return dist.neg();
    }
}