package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class NormalizeRotation implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = data.pos;
        // 1. 中心化
        pos = pos.sub(pos.mean(new long[]{0}, true, new ScalarTypeOptional(kFloat())));

        // 2. 计算协方差矩阵 C = (X^T * X) / (N-1)
        Tensor cov = pos.t().mm(pos).div(new Tensor(pos.size(0) - 1));

        // 3. 特征值分解 (Eigenvalue decomposition)
        // result[0] 为特征值，result[1] 为特征向量
        T_TensorTensor_T eig = linalg_eig(cov);
        Tensor v = eig.get1().to(kFloat()); // 取出旋转矩阵

        // 4. 应用旋转
        data.pos = pos.mm(v);
        return data;
    }
}