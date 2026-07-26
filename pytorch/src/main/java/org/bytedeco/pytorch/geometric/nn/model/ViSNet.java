package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
public class ViSNet extends Module {
    // 简化版：仅演示 Vector-Scalar 交互逻辑
    private LinearImpl s2v;
    private LinearImpl v2s;

    public ViSNet(long hiddenDim) {
        this.s2v = new LinearImpl(hiddenDim, hiddenDim);
        this.v2s = new LinearImpl(hiddenDim, hiddenDim);
        register_module("s2v", s2v);
        register_module("v2s", v2s);
    }

    /**
     * @param s 标量特征 [N, C]
     * @param v 向量特征 [N, 3, C]
     */
    public Tensor[] forwardVisNet(Tensor s, Tensor v) {
        // 1. Scalar to Vector Influence
        // scale = MLP(s)
        Tensor scale = s2v.forward(s).unsqueeze(1); // [N, 1, C]
        // v_new = v * scale
        Tensor vNew = v.mul(scale);

        // 2. Vector to Scalar Influence (via Dot Product / Norm)
        // norm = ||v||^2 -> [N, C]
        Tensor vNorm = v.pow(new Scalar(2)).sum(new long[]{1}, false,new ScalarTypeOptional(torch.ScalarType.Float));
        Tensor sUpdate = v2s.forward(vNorm);
        Tensor sNew = s.add(sUpdate);

        return new Tensor[]{sNew, vNew};
    }
}