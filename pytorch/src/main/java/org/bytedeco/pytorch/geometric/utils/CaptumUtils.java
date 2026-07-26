package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.*;

public class CaptumUtils {

    /**
     * 准备 Captum 输入：为节点特征开启梯度追踪
     */
    public static Tensor to_captum_input(Tensor x) {
        // Clone detach to ensure leaf
        Tensor input = x.clone().detach();
        input.set_requires_grad(true);
        return input;
    }

    /**
     * 转换 Mask 类型 (PyG style mask to Indices)
     */
    public static Tensor mask_to_indices(Tensor mask) {
        return mask.nonzero().squeeze(1);
    }
}