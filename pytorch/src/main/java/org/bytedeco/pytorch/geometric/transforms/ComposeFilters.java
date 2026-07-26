package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.geometric.data.GraphData;

import java.util.List;

/**
 * ComposeFilters: 组合多个过滤器
 * 只有当所有内部过滤器都返回 true 时，该数据才会被保留
 */
public class ComposeFilters {
    private List<GraphFilter> filters;

    public ComposeFilters(List<GraphFilter> filters) {
        this.filters = filters;
    }

    public boolean apply(GraphData data) {
        for (GraphFilter f : filters) {
            if (!f.filter(data)) {
                return false;
            }
        }
        return true;
    }
}