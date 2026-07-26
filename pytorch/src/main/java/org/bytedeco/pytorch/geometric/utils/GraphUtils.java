package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class GraphUtils {

    /**
     * 为 edge_index 添加自环: A_hat = A + I
     * * @param edge_index 原始边索引 [2, E]
     * @param numNodes   节点数量 N
     * @return 拼接后的边索引 [2, E + N]
     */
    public static Tensor add_self_loops(Tensor edge_index, long numNodes) {
        // 1. 创建自环索引 [0, 1, ..., N-1]
        // arange 生成 [N]，然后 repeat 两次变成 [2, N]
        Tensor loop_index = torch.arange(new Scalar(0), new Scalar(numNodes), edge_index.options());

        // unsqueeze(0) 将 [N] 变成 [1, N]
        // repeat({2, 1}) 将 [1, N] 变成 [2, N]
        loop_index = loop_index.unsqueeze(0).repeat(new long[]{2, 1});

        // 2. 将原始 edge_index 与 loop_index 在维度 1 (列方向) 拼接
        // cat 接收 Tensor 数组，dimension 设为 1
        return torch.cat(new TensorVector(edge_index, loop_index), 1);
    }
}
