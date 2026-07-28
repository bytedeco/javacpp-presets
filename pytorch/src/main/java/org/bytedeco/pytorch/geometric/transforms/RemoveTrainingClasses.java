/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.RemoveTrainingClasses
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.logical_and;
import static org.bytedeco.pytorch.global.torch.logical_not;
import static org.bytedeco.pytorch.global.torch.ones;

/**
 * Zero-out {@code train_mask} entries whose label is in {@code classesToRemove}.
 * If {@code train_mask} is absent, creates an all-true mask first (demo-friendly).
 */
public class RemoveTrainingClasses implements BaseTransform {
    private final List<Integer> classesToRemove;

    public RemoveTrainingClasses(List<Integer> classesToRemove) {
        if (classesToRemove == null) throw new NullPointerException("classesToRemove");
        this.classesToRemove = new ArrayList<>(classesToRemove);
    }

    @Override
    public GraphData apply(GraphData data) {
        TransformUtils.requireData(data);
        if (data.y == null || !data.y.defined()) {
            throw new IllegalArgumentException("RemoveTrainingClasses requires data.y");
        }
        long n = data.y.size(0);
        Tensor trainMask = data.get("train_mask");
        if (trainMask == null || !trainMask.defined()) {
            trainMask = ones(new long[]{n}, TransformUtils.boolOptsLike(data.y));
            data.put("train_mask", trainMask);
        }
        for (Integer cls : classesToRemove) {
            Tensor classMask = data.y.eq(new Scalar(cls.intValue()));
            trainMask = logical_and(trainMask, logical_not(classMask));
        }
        data.put("train_mask", trainMask);
        return data;
    }
}
