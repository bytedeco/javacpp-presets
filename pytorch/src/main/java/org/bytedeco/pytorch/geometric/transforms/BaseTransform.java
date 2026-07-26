package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.geometric.data.GraphData;


/**
 * BaseTransform: 抽象基类
 * public abstract class BaseTransform {
 *     public abstract GraphData apply(GraphData data);
 * }
 */
@FunctionalInterface
public interface BaseTransform {
    
      GraphData apply(GraphData data);

    /**
     * Chains two transforms together, similar to T.Compose
     */
    default BaseTransform andThen(BaseTransform next) {
        return (GraphData data) -> next.apply(this.apply(data));
    }
}