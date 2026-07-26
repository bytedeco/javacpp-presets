package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * NodePropertySplit: 基于节点属性的分布偏移划分
 * 原理：根据某个属性（如节点的度、金额、年龄）排序，取不同区间的节点作为训练/验证/测试集。
 */
public class NodePropertySplit implements BaseTransform {
    private double trainRatio, valRatio;
    private boolean ascending;

    public NodePropertySplit(double trainRatio, double valRatio, boolean ascending) {
        this.trainRatio = trainRatio;
        this.valRatio = valRatio;
        this.ascending = ascending;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor prop = data.x.select(1, 0);
        long numNodes = data.numNodes();

        Tensor sortedIndices = sort(prop, 0, ascending).get1();

        long nTrain = (long) (numNodes * trainRatio);
        long nVal = (long) (numNodes * valRatio);

        // 1. 初始化 Mask 为 false
        Tensor trainMask = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool())));
        Tensor valMask = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool())));
        Tensor testMask = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool())));

        // 2. 提取切片索引
        Tensor trainIdx = sortedIndices.slice(0, new LongOptional(0), new LongOptional(nTrain), 1);
        Tensor valIdx = sortedIndices.slice(0, new LongOptional(nTrain), new LongOptional(nTrain + nVal), 1);
        Tensor testIdx = sortedIndices.slice(0, new LongOptional(nTrain + nVal), new LongOptional(numNodes), 1);

        // 3. 关键点：使用 index_fill_ 将对应索引设为 1 (即 true)
        // 这是最稳妥的方法，避开了 tensor(true) 的类型 Bug
        trainMask.index_fill_(0, trainIdx, new Scalar(1));
        valMask.index_fill_(0, valIdx, new Scalar(1));
        testMask.index_fill_(0, testIdx, new Scalar(1));

        // 4. 存入数据对象
        data.put("train_mask", trainMask);
        data.put("val_mask", valMask);
        data.put("test_mask", testMask);

        return data;
    }
//    @Override
    public GraphData call2(GraphData data) {
        // 1. 获取用于排序的属性 (假设存放在 data.node_prop 中)
        // 例如在风控中，这是用户的"历史交易总额"
//        Tensor prop = data.node_prop;
        Tensor prop = data.x.select(1, 0);
        long numNodes =data.numNodes(); //prop.size(0);

        // 2. 排序获取索引
        Tensor sortedIndices = sort(prop, 0, ascending).get1();

        // 3. 按比例切分区间 (产生分布偏移) // data.numNodes();// 
        long nTrain = (long) (numNodes * trainRatio);
        long nVal = (long) (numNodes * valRatio);

        data.put("train_mask", zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool()))));
        data.put("val_mask", zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool()))));
        data.put("test_mask", zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kBool()))));

        // 训练集取属性较小的一端，测试集取属性较大的一端
        Tensor trainIndices = sortedIndices.slice(0, new LongOptional(0), new LongOptional(nTrain),1);
        Tensor trainMask = data.get("train_mask");
        Tensor tr = tensor(true, trainMask.options().dtype(new ScalarTypeOptional(kBool())));
        trainMask.index_put_( new TensorIndexVector(new TensorIndex(trainIndices)),tr );
        Tensor valIndices = sortedIndices.slice(0, new LongOptional(nTrain), new LongOptional(nTrain + nVal),1);
        Tensor valMask = data.get("val_mask");
        Tensor vr = tensor(true, valMask.options());
        valMask.index_put_(new TensorIndexVector(new TensorIndex(valIndices)), vr);
        
        Tensor testIndices = sortedIndices.slice(0, new LongOptional(nTrain + nVal), new LongOptional(numNodes),1);
        Tensor testMask = data.get("test_mask");
        Tensor ttr = tensor(true, testMask.options());
        testMask.index_put_(new TensorIndexVector(new TensorIndex(testIndices)), ttr);
        
//        data.get("train_mask").index_put_(new Tensor[]{}, tensor(true, data.get("train_mask").options()));
//        data.get("val_mask").index_put_(new Tensor[]{}, tensor(true, data.get("val_mask").options()));
//        data.get("test_mask").index_put_(new Tensor[]{sortedIndices.slice(0, new LongOptional(nTrain + nVal), new LongOptional(numNodes),1)}, tensor(true, data.get("test_mask").options()));

        return data;
    }
}