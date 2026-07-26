package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public class Spherical implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor row = data.edge_index.select(0, 0);
        Tensor col = data.edge_index.select(0, 1);

        Tensor v = data.pos.index_select(0, col).sub(data.pos.index_select(0, row));

        Tensor x = v.select(1, 0);
        Tensor y = v.select(1, 1);
        Tensor z = v.select(1, 2);

        Tensor r = v.pow(new Scalar(2)).sum(1).sqrt();
        // phi = atan2(y, x)
        Tensor phi = atan2(y, x).div(new Scalar(2 * Math.PI)).add(new Scalar(0.5));
        // theta = acos(z / r)
        Tensor theta = acos(z.div(r.add(new Scalar(1e-7)))).div(new Scalar(Math.PI));

        Tensor spherical = stack(new TensorVector(r, phi, theta), 1);

        data.edge_attr = (data.edge_attr == null) ? spherical : cat(new TensorVector(data.edge_attr, spherical), 1);
        return data;
    }
}