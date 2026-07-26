package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * RandomNodeSplit: 节点级随机划分
 * 为 Data 对象添加 train_mask, val_mask, test_mask
 */
public class RandomNodeSplit implements BaseTransform {
    private double trainRatio, valRatio, testRatio;

    public RandomNodeSplit(double trainRatio, double valRatio, double testRatio) {
        this.trainRatio = trainRatio;
        this.valRatio = valRatio;
        this.testRatio = testRatio;
    }

    @Override
    public GraphData apply(GraphData data) {
        long numNodes = data.numNodes();

        // 1. 生成随机排列索引
        Tensor perm = randperm(numNodes, data.x.options().dtype(new ScalarTypeOptional(kLong())));

        long trainSize = (long) (numNodes * trainRatio);
        long valSize = (long) (numNodes * valRatio);

        // 2. 构造 Mask (默认为 false / 0)
        Tensor trainMask = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool())));
        Tensor valMask = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool())));
        Tensor testMask = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool())));

        // 3. 使用 index_fill_ 填充 Mask
        // index_fill_(维度, 索引Tensor, 填充值)
        // 这里的 1 会被自动转为 Bool 类型的 true
        trainMask.index_fill_(0, perm.narrow(0, 0, trainSize), new Scalar(1));
        valMask.index_fill_(0, perm.narrow(0, trainSize, valSize), new Scalar(1));
        testMask.index_fill_(0, perm.narrow(0, trainSize + valSize, numNodes - (trainSize + valSize)),new Scalar(1) );

        // 4. 将结果存入 GraphData
        data.put("train_mask", trainMask);
        data.put("val_mask", valMask);
        data.put("test_mask", testMask);

        return data;
    }
    
//    @Override
    public GraphData call2(GraphData data) {
        long numNodes = data.x.size(0);
        // 生成随机排列索引
        Tensor perm = randperm(numNodes, data.x.options().dtype(new ScalarTypeOptional(kLong())));

        long trainSize = (long) (numNodes * trainRatio);
        long valSize = (long) (numNodes * valRatio);

        // 构造 Mask (默认为 false)
        Tensor train_mask = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool())));
        Tensor val_mask = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool())));
        Tensor test_mask = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool())));

        // 根据随机索引填充 Mask
        train_mask.index_put_(new TensorIndexVector(new TensorIndex(perm.narrow(0, 0, trainSize))), tensor(true, data.x.options().dtype(new ScalarTypeOptional(kBool())) ) );
        val_mask.index_put_(new TensorIndexVector(new TensorIndex(perm.narrow(0, trainSize, valSize))), tensor(true, data.x.options().dtype(new ScalarTypeOptional(kBool()))));
        test_mask.index_put_(new TensorIndexVector(new TensorIndex(perm.narrow(0, trainSize + valSize, numNodes - (trainSize + valSize)))), tensor(true, data.x.options().dtype(new ScalarTypeOptional(kBool()))) );

        return data;
    }
}