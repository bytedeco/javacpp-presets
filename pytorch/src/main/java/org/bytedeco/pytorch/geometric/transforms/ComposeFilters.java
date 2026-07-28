/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.ComposeFilter
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.geometric.data.GraphData;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * AND-compose graph filters — data is kept only if every filter returns true.
 */
public class ComposeFilters implements GraphFilter {

    private final List<GraphFilter> filters;

    public ComposeFilters(List<GraphFilter> filters) {
        if (filters == null) {
            throw new NullPointerException("filters");
        }
        this.filters = Collections.unmodifiableList(new ArrayList<>(filters));
    }

    public ComposeFilters(GraphFilter... filters) {
        this(Arrays.asList(filters));
    }

    /** Keep the historical {@code apply} name used by demos. */
    public boolean apply(GraphData data) {
        return filter(data);
    }

    @Override
    public boolean filter(GraphData data) {
        TransformUtils.requireData(data);
        for (GraphFilter f : filters) {
            if (f == null) {
                throw new NullPointerException("ComposeFilters contains a null filter");
            }
            if (!f.filter(data)) {
                return false;
            }
        }
        return true;
    }
}
