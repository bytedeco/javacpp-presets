package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.geometric.data.GraphData;

/**
 * Filter 接口：用于定义图数据的过滤规则
 */
public interface GraphFilter {
    boolean filter(GraphData data);
}