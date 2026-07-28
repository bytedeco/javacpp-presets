/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.SamplePoints
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.GeneratorOptional;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/** Area-weighted barycentric sampling of {@code num} points on a triangle mesh. */
public class SamplePoints implements BaseTransform {
    private final int num;
    public SamplePoints(int num) {
        if (num <= 0) throw new IllegalArgumentException("num must be > 0");
        this.num = num;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = TransformUtils.requirePos(data);
        Tensor face = data.get("face");
        if (face == null || !face.defined()) {
            throw new IllegalArgumentException("SamplePoints requires data['face']");
        }
        Tensor v1 = pos.index_select(0, face.select(0, 0));
        Tensor v2 = pos.index_select(0, face.select(0, 1));
        Tensor v3 = pos.index_select(0, face.select(0, 2));
        Tensor areas = cross(v2.sub(v1), v3.sub(v1), new LongOptional(1))
                .norm(new ScalarOptional(new Scalar(2)), 1)
                .mul(new Scalar(0.5f))
                .clamp_min(new Scalar(1e-12));
        Tensor faceIdx = multinomial(areas, num, true, new GeneratorOptional());
        Tensor r1 = rand(new long[]{num, 1}, pos.options()).sqrt();
        Tensor r2 = rand(new long[]{num, 1}, pos.options());
        Tensor p1 = v1.index_select(0, faceIdx).mul(ones_like(r1).sub(r1));
        Tensor p2 = v2.index_select(0, faceIdx).mul(r1.mul(ones_like(r2).sub(r2)));
        Tensor p3 = v3.index_select(0, faceIdx).mul(r1.mul(r2));
        data.pos = p1.add(p2).add(p3);
        // mesh topology no longer valid
        data.edge_index = null;
        data.x = null; // sampled cloud — drop vertex features
        return data;
    }
}
