package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

// Decoder
class InnerProductDecoder extends Module {

    public Tensor forward(Tensor z, Tensor edge_index, boolean sigmoid) {
        // value = (z[u] * z[v]).sum()
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        Tensor zRow = z.index_select(0, row);
        Tensor zCol = z.index_select(0, col);

        Tensor value = zRow.mul(zCol).sum(new long[]{1}, false,new ScalarTypeOptional());
        return sigmoid ? torch.sigmoid(value) : value;
    }
}

