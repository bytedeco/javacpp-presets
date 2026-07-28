/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.Compose
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.geometric.data.GraphData;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Sequentially apply a list of transforms (PyG {@code T.Compose}).
 */
public class Compose implements BaseTransform {

    private final List<BaseTransform> transforms;

    public Compose(List<BaseTransform> transforms) {
        if (transforms == null) {
            throw new NullPointerException("transforms");
        }
        this.transforms = Collections.unmodifiableList(new ArrayList<>(transforms));
    }

    public Compose(BaseTransform... transforms) {
        this(Arrays.asList(transforms));
    }

    @Override
    public GraphData apply(GraphData data) {
        TransformUtils.requireData(data);
        for (BaseTransform t : transforms) {
            if (t == null) {
                throw new NullPointerException("Compose contains a null transform");
            }
            data = t.apply(data);
            if (data == null) {
                throw new IllegalStateException(
                        t.getClass().getSimpleName() + ".apply returned null");
            }
        }
        return data;
    }

    public List<BaseTransform> getTransforms() {
        return transforms;
    }

    @Override
    public String toString() {
        return "Compose(" + transforms + ")";
    }
}
