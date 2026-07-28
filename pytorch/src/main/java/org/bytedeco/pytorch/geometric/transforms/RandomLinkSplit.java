/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.RandomLinkSplit
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.randperm;

/**
 * Random edge split into train/val/test edge_index stores
 * ({@code train_edge_index}, {@code val_edge_index}, {@code test_edge_index}).
 */
public class RandomLinkSplit implements BaseTransform {
    private final double numVal, numTest;

    public RandomLinkSplit(double numVal, double numTest) {
        if (numVal < 0 || numTest < 0 || numVal + numTest >= 1.0) {
            throw new IllegalArgumentException(
                    "numVal/numTest must be >=0 and sum < 1");
        }
        this.numVal = numVal;
        this.numTest = numTest;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        long numEdges = ei.size(1);
        Tensor perm = randperm(numEdges, TransformUtils.longOptsLike(ei));
        long nVal = (long) (numEdges * numVal);
        long nTest = (long) (numEdges * numTest);
        long nTrain = numEdges - nVal - nTest;
        if (nTrain < 0) nTrain = 0;

        data.put("train_edge_index", ei.index_select(1, perm.narrow(0, 0, nTrain)));
        data.put("val_edge_index", ei.index_select(1, perm.narrow(0, nTrain, nVal)));
        data.put("test_edge_index",
                ei.index_select(1, perm.narrow(0, nTrain + nVal, nTest)));
        return data;
    }
}
