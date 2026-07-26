package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.linalg_svd;

/**
 * SVDFeatureReduction: 通过 SVD 进行特征降维
 */
public class SVDFeatureReduction implements BaseTransform {
    private int outChannels;
    public SVDFeatureReduction(int outChannels) { this.outChannels = outChannels; }

    @Override
    public GraphData apply(GraphData data) {
        // 执行 SVD: X = U * S * V^T
        T_TensorTensorTensor_T svd = linalg_svd(data.x, false, new StringViewOptional());
        Tensor u = svd.get0(); // [N, N]
        Tensor s = svd.get1(); // [N]

        // 投影到低维空间: U_reduced * S_reduced
        data.x = u.narrow(1, 0, outChannels).mul(s.narrow(0, 0, outChannels));
        return data;
    }
}