/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.RandomNodeSplit
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/** Random train/val/test node masks. */
public class RandomNodeSplit implements BaseTransform {
    private final double trainRatio, valRatio, testRatio;

    public RandomNodeSplit(double trainRatio, double valRatio, double testRatio) {
        double sum = trainRatio + valRatio + testRatio;
        if (trainRatio < 0 || valRatio < 0 || testRatio < 0 || Math.abs(sum - 1.0) > 1e-6) {
            throw new IllegalArgumentException(
                    "ratios must be non-negative and sum to 1, got "
                            + trainRatio + "+" + valRatio + "+" + testRatio);
        }
        this.trainRatio = trainRatio;
        this.valRatio = valRatio;
        this.testRatio = testRatio;
    }

    @Override
    public GraphData apply(GraphData data) {
        long numNodes = TransformUtils.numNodes(data);
        Tensor ref = data.x != null ? data.x : data.pos;
        Tensor perm = randperm(numNodes, TransformUtils.longOptsLike(ref));
        long trainSize = (long) (numNodes * trainRatio);
        long valSize = (long) (numNodes * valRatio);
        long testSize = numNodes - trainSize - valSize;

        Tensor trainMask = zeros(new long[]{numNodes}, TransformUtils.boolOptsLike(ref));
        Tensor valMask = zeros(new long[]{numNodes}, TransformUtils.boolOptsLike(ref));
        Tensor testMask = zeros(new long[]{numNodes}, TransformUtils.boolOptsLike(ref));

        if (trainSize > 0) trainMask.index_fill_(0, perm.narrow(0, 0, trainSize), new Scalar(1));
        if (valSize > 0) valMask.index_fill_(0, perm.narrow(0, trainSize, valSize), new Scalar(1));
        if (testSize > 0) testMask.index_fill_(0, perm.narrow(0, trainSize + valSize, testSize), new Scalar(1));

        data.put("train_mask", trainMask);
        data.put("val_mask", valMask);
        data.put("test_mask", testMask);
        return data;
    }
}
