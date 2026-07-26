package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

public class CorrectAndSmooth {

    /**
     * Label Propagation: Y = alpha * D^-1/2 A D^-1/2 Y + (1-alpha) Y_init
     */
    public static Tensor propagate(Tensor y, Tensor edge_index, double alpha, int numProp) {
        long N = y.size(0);
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        // Symmetric Norm
        Tensor deg = AggrUtils.compute_degree(row, N);
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));
        Tensor norm = degInvSqrt.index_select(0, row).mul(degInvSqrt.index_select(0, col));

        Tensor out = y;
        Tensor yInit = y;

        for (int i = 0; i < numProp; i++) {
            // Message passing
            Tensor x_j = out.index_select(0, col);
            Tensor msg = x_j.mul(norm.unsqueeze(1));
            Tensor aggr = AggrUtils.scatter(msg, row, N, "sum");

            // Update
            out = aggr.mul(new Scalar(alpha)).add(yInit.mul(new Scalar(1 - alpha)));
        }
        return out;
    }

    /**
     * Correct step: Propagate training errors
     */
    public static Tensor correct(Tensor ySoft, Tensor yTrue, Tensor mask, Tensor edge_index, double alpha, int numProp) {
        // Error: E = Y_true - Y_soft (only on training nodes)
        Tensor error = torch.zeros_like(ySoft);
        error.index_put_(new TensorIndexVector(mask), yTrue.index_select(0, mask).sub(ySoft.index_select(0, mask)));

        // Propagate Error
        Tensor smoothedError = propagate(error, edge_index, alpha, numProp);

        // Y_new = Y_soft + Scale * SmoothedError
        return ySoft.add(smoothedError);
    }

    /**
     * Smooth step: Propagate predictions
     */
    public static Tensor smooth(Tensor yCorrected, Tensor yTrue, Tensor mask, Tensor edge_index, double alpha, int numProp) {
        Tensor yIn = yCorrected.clone();
        // Replace training nodes with ground truth
        yIn.index_put_(new TensorIndexVector(mask), yTrue.index_select(0, mask));

        return propagate(yIn, edge_index, alpha, numProp);
    }
}