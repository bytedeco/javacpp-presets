package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.kLong;
import static org.bytedeco.pytorch.global.torch.randperm;

/**
 * RandomLinkSplit: 边级随机划分（用于链路预测任务）
 * 将边划分为训练、验证、测试集，并生成负采样边
 */
public class RandomLinkSplit implements BaseTransform {
    private double numVal, numTest;

    public RandomLinkSplit(double numVal, double numTest) {
        this.numVal = numVal;
        this.numTest = numTest;
    }

    @Override
    public GraphData apply(GraphData data) {
        long numEdges = data.edge_index.size(1);
        Tensor perm = randperm(numEdges, data.edge_index.options().dtype(new ScalarTypeOptional(kLong())));

        long nVal = (long) (numEdges * numVal);
        long nTest = (long) (numEdges * numTest);
        long nTrain = numEdges - nVal - nTest;

        // 划分训练边、验证边、测试边
        data.put("train_edge_index", data.edge_index.index_select(1, perm.narrow(0, 0, nTrain)));
        data.put("val_edge_index", data.edge_index.index_select(1, perm.narrow(0, nTrain, nVal)));
        data.put("test_edge_index", data.edge_index.index_select(1, perm.narrow(0, nTrain + nVal, nTest)));

        return data;
    }
}