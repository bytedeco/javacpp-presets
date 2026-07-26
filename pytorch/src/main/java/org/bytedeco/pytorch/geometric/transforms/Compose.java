package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.geometric.data.GraphData;

import java.util.List;

/**
 * Compose: 将多个变换组合在一起
 */
public class Compose implements BaseTransform {
    private List<BaseTransform> transforms;

    public Compose(List<BaseTransform> transforms) {
        this.transforms = transforms;
    }

    @Override
    public GraphData apply(GraphData data) {
        for (BaseTransform t : transforms) {
            data = t.apply(data);
        }
        return data;
    }
}
