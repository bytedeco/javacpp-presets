/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.BaseTransform
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.geometric.data.GraphData;

/**
 * Functional transform over a single {@link GraphData} sample
 * (PyG {@code BaseTransform} / {@code T.BaseTransform}).
 *
 * <p>Implementations should:
 * <ul>
 *   <li>validate required fields via {@link TransformUtils}</li>
 *   <li>mutate and return the same {@code data} instance (PyG in-place style),
 *       unless a new graph is intentionally produced (e.g. pooling)</li>
 *   <li>never return {@code null}</li>
 * </ul>
 */
@FunctionalInterface
public interface BaseTransform {

    GraphData apply(GraphData data);

    /** Chain {@code this} then {@code next} (PyG {@code T.Compose([this, next])}). */
    default BaseTransform andThen(BaseTransform next) {
        if (next == null) {
            throw new NullPointerException("next");
        }
        return (GraphData data) -> next.apply(this.apply(data));
    }
}
