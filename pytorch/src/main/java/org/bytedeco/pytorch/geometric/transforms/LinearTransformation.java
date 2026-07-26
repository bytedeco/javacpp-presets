package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

public  class LinearTransformation implements BaseTransform {
    private final Tensor matrix;

    public LinearTransformation(Tensor matrix) { this.matrix = matrix; }

    @Override
    public GraphData apply(GraphData data) {
        // data.pos: [N, D], matrix: [D, D]
        data.pos = data.pos.mm(matrix);
        return data;
    }
}
