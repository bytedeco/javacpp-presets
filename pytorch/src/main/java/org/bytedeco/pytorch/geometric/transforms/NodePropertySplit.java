/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.NodePropertySplit
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.sort;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Sort nodes by a scalar property and split into train/val/test by rank.
 * Property source (in order): {@code data['node_prop']}, else {@code x[:,0]}.
 */
public class NodePropertySplit implements BaseTransform {
    private final double trainRatio, valRatio;
    private final boolean ascending;

    public NodePropertySplit(double trainRatio, double valRatio, boolean ascending) {
        if (trainRatio < 0 || valRatio < 0 || trainRatio + valRatio >= 1.0) {
            throw new IllegalArgumentException("invalid ratios");
        }
        this.trainRatio = trainRatio;
        this.valRatio = valRatio;
        this.ascending = ascending;
    }

    @Override
    public GraphData apply(GraphData data) {
        long numNodes = TransformUtils.numNodes(data);
        Tensor prop = data.get("node_prop");
        if (prop == null || !prop.defined()) {
            Tensor x = TransformUtils.requireX(data);
            prop = x.select(1, 0);
        }
        // sort(..., descending = !ascending)
        T_TensorTensor_T sorted = sort(prop, 0, !ascending);
        Tensor sortedIndices = sorted.get1();

        long nTrain = (long) (numNodes * trainRatio);
        long nVal = (long) (numNodes * valRatio);
        long nTest = numNodes - nTrain - nVal;

        Tensor ref = data.x != null ? data.x : prop;
        Tensor trainMask = zeros(new long[]{numNodes}, TransformUtils.boolOptsLike(ref));
        Tensor valMask = zeros(new long[]{numNodes}, TransformUtils.boolOptsLike(ref));
        Tensor testMask = zeros(new long[]{numNodes}, TransformUtils.boolOptsLike(ref));

        if (nTrain > 0) {
            trainMask.index_fill_(0, sortedIndices.slice(0, new LongOptional(0), new LongOptional(nTrain), 1), new Scalar(1));
        }
        if (nVal > 0) {
            valMask.index_fill_(0, sortedIndices.slice(0, new LongOptional(nTrain), new LongOptional(nTrain + nVal), 1), new Scalar(1));
        }
        if (nTest > 0) {
            testMask.index_fill_(0, sortedIndices.slice(0, new LongOptional(nTrain + nVal), new LongOptional(numNodes), 1), new Scalar(1));
        }
        data.put("train_mask", trainMask);
        data.put("val_mask", valMask);
        data.put("test_mask", testMask);
        return data;
    }
}
