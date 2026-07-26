package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
/**
 * PANPooling: 基于路径积分的池化
 * 核心：通过拉普拉斯能量或路径权重选择得分最高的节点
 */
public class PANPooling extends Module {
    private LinearImpl scoreLayer;
    private double ratio;

    public PANPooling(long inDim, double ratio) {
        this.ratio = ratio;
        this.scoreLayer = register_module("scoreLayer", new LinearImpl(inDim, 1));
    }

    public Tensor[] panPool(Tensor x, Tensor edge_index) {
        // 1. 计算每个节点的得分
        Tensor score = scoreLayer.forward(x).sigmoid().view(-1);

        // 2. 根据比例选择保留的节点索引 (Top-K)
        long numNodes = x.size(0);
        long k = Math.max(1, (long)(numNodes * ratio));

        T_TensorTensor_T valuesIndices = torch.topk(score, k);
        Tensor perm = valuesIndices.get1(); // 获取 Top-K 的索引

        // 3. 抽取节点特征
        Tensor x_pooled = x.index_select(0, perm);

        // 4. 调整得分权重
        x_pooled = x_pooled.mul(score.index_select(0, perm).view(-1, 1));

        return new Tensor[]{x_pooled, perm}; // 返回池化后的特征和保留的索引
    }
}
