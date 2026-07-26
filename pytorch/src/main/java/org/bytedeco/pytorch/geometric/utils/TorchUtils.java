package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

public class TorchUtils {
    // 专门治愈 torch.cat 的语法糖
    public static Tensor cat(Tensor t1, Tensor t2, long dim) {
        return torch.cat(new TensorVector(t1, t2), dim);
    }

    // 专门治愈 数值计算
    public static Scalar s(double val) {
        return new Scalar(val);
    }
}